import torch
from sklearn.metrics import f1_score
from torch.nn import functional as torch_fn
from transformers import DataCollatorWithPadding

from qa_nlp.dataset import create_QC_dataset, load
from qa_nlp.models import QCModel, get_tokenizer
from argument_parser import ArgumentParser

from qa_nlp.utils import json2dict, load_class, get_gradient_accumulate_steps
from scheduler import WarmupLinearLR

from math import ceil

arg_parser = ArgumentParser()

arg_parser.define("EPOCHS", default=20, arg_type=int)
arg_parser.define("lr", default=2e-5, arg_type=float)
arg_parser.define("weight-decay", default=0.001, arg_type=float)
arg_parser.define("use-gpu", default=False, arg_type=bool)
arg_parser.define("warmup-rate", default=0.1, arg_type=float)
arg_parser.define("gradient-accumulation-steps", default=5, arg_type=int)
arg_parser.define("eval-steps", default=5, arg_type=int)
arg_parser.define("patience-steps", default=20, arg_type=int)
arg_parser.define("max-grad-norm", default=1.0, arg_type=float)
arg_parser.define("save-checkpoint", default=True, arg_type=bool)
arg_parser.define("model-name", default="qc/V2023-10-23", arg_type=str)

flags = arg_parser.parse()


def evaluate(model, val_dataloader, val_steps, device):
	model.eval()
	with torch.no_grad():
		total_loss = 0.0
		y_trues = []
		y_preds = []
		
		for batch in val_dataloader:
			inputs = {'input_ids': batch["input_ids"].to(device),
			          'attention_mask': batch["attention_mask"].to(device),
			          'token_type_ids': batch["token_type_ids"].to(device)}
			
			logits = model(inputs)
			predicts = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
			y_preds.extend(predicts)
			y_trues.extend(batch["labels"].numpy().tolist())
			
			loss = torch_fn.cross_entropy(logits, batch["labels"].to(device))
			total_loss += loss.item()
		
		f1_score_micro = f1_score(y_trues, y_preds, average="micro")
		f1_score_macro = f1_score(y_trues, y_preds, average="macro")
		
		validation_result = {
			"loss": round(total_loss / val_steps, 4),
			"macro_f1": round(f1_score_macro, 4),
			"micro_f1": round(f1_score_micro, 4)
		}
	
	return validation_result


if __name__ == "__main__":
	device = torch.device("cuda" if flags.use_gpu and torch.cuda.is_available() else "cpu")
	print(flags.use_gpu)
	class_retriever = load_class()
	class2index = class_retriever["class2index"]
	index2class = class_retriever["index2class"]
	
	tokenizer = get_tokenizer("bert-base-multilingual-cased")
	train_dataset = create_QC_dataset("data/question_classification/vie/train.txt",
	                                  tokenizer,
	                                  max_length=512,
	                                  class2index=class2index)
	val_dataset = create_QC_dataset("data/question_classification/vie/test.txt",
	                                tokenizer,
	                                max_length=512,
	                                class2index=class2index)
	collator = DataCollatorWithPadding(tokenizer)
	train_dataloader = load(train_dataset, collator, batch_size=8)
	val_dataloader = load(val_dataset, collator, batch_size=8)
	
	config = json2dict("assets/model_params/qc-model.json")
	model = QCModel(**config).to(device)
	
	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		 'weight_decay': flags.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
		 'weight_decay': 0.0}
	]
	
	total_steps = flags.EPOCHS * ceil(len(train_dataloader) / flags.gradient_accumulation_steps)
	warmup_steps = int(0.1 * total_steps)
	monitor_f1 = float('-inf')
	
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=flags.lr, weight_decay=0.0)
	scheduler = WarmupLinearLR(optimizer, warmup_steps, total_steps, min_proportion=0.01)
	steps_per_epoch = len(train_dataloader)
	global_steps = 0
	temp_patience = flags.patience_steps
	
	log_writer = open("logs/train-qc.log", "w")
	log_writer.write("               ***** Start training *****\n")
	log_writer.write("============================================================\n")
	log_writer.write(f"Num samples: {len(train_dataset)}\n")
	log_writer.write(f"Num epochs: {flags.EPOCHS}\n")
	log_writer.write(f"Gradient accumulation steps = {flags.gradient_accumulation_steps}\n")
	log_writer.write("============================================================\n")
	
	for epoch in range(flags.EPOCHS):
		total_loss = 0.0
		y_trues = []
		y_preds = []
		
		log_writer.write("------------------------------------------------------------\n")
		log_writer.write(f"Epoch {epoch + 1:>3d}/{flags.EPOCHS}:\n")
		
		print(f"Epoch \033[92m{epoch + 1:>3d}/{flags.EPOCHS}\033[00m:")
		
		for step, batch in enumerate(train_dataloader):
			print(f"\r- Step \033[96m{step + 1:>5d}/{steps_per_epoch}\033[00m:", end="")
			
			curr_acc_step = get_gradient_accumulate_steps(steps_per_epoch, step, flags.gradient_accumulation_steps)
			model.train()
			
			inputs = {'input_ids': batch["input_ids"].to(device),
			          'attention_mask': batch["attention_mask"].to(device),
			          'token_type_ids': batch["token_type_ids"].to(device)}
			
			logits = model(inputs)
			predicts = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
			y_preds.extend(predicts)
			y_trues.extend(batch["labels"].numpy().tolist())
			
			loss = torch_fn.cross_entropy(logits, batch["labels"].to(device))
			total_loss += loss.item()
			loss /= curr_acc_step
			loss.backward()
			
			if (step + 1) % flags.gradient_accumulation_steps == 0 or (step == steps_per_epoch - 1):
				torch.nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
				
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				optimizer.zero_grad()
				global_steps += 1
				
				if global_steps % flags.eval_steps == 0 or (step == steps_per_epoch - 1):
					print()
					logging_line = f"- Step: {step + 1:>5d}/{steps_per_epoch}, lr: {scheduler.get_last_lr()}\n"
					log_writer.write(logging_line)
					
					f1_score_micro = f1_score(y_trues, y_preds, average="micro")
					f1_score_macro = f1_score(y_trues, y_preds, average="macro")
					
					train_accumulate_loss = round(total_loss / (step + 1), 4)
					train_accumulate_macro_f1 = round(f1_score_macro, 4)
					train_accumulate_micro_f1 = round(f1_score_micro, 4)
					
					train_result_line = (f"{'loss':8s}: {train_accumulate_loss:<10.4f} - "
					                     f"{'macro_f1':12s}: {train_accumulate_macro_f1:<10.4f} - "
					                     f"{'micro_f1':12s}: {train_accumulate_micro_f1:<10.4f}")
					
					print(f"    \033[95m{'Train result':20s}\033[00m - {train_result_line}")
					log_writer.write(f"    {'Train result':20s} - {train_result_line}\n")
					
					validation_output = evaluate(model, val_dataloader, len(val_dataloader), device)
					
					val_result_line = (f"val_loss: {validation_output['loss']:<10.4f} - "
					                   f"val_macro_f1: {validation_output['macro_f1']:<10.4f} - "
					                   f"val_micro_f1: {validation_output['micro_f1']:<10.4f}")
					
					print(f"    \033[95m{'Validation result':20s}\033[00m - {val_result_line}")
					log_writer.write(f"    {'Validation result':20s} - {val_result_line}\n")
					if flags.save_checkpoint:
						if validation_output['micro_f1'] > monitor_f1:
							model.save(flags.model_name)
							log_writer.write(
								f"    # val_f1 improve from {monitor_f1} to {validation_output['micro_f1']}. "
								f"Saving model with name \"{flags.model_name}\"")
							monitor_f1 = validation_output["micro_f1"]
							temp_patience = flags.patience_steps
						else:
							temp_patience -= 1
					log_writer.write("\n")
				
			
			if flags.patience_steps > 0 and temp_patience == 0:
				log_writer.write("============================================================\n")
				log_writer.write(f"                    # Early Stopped\n")
				log_writer.write("============================================================\n")
				break
		
		else:
			continue
		
		break
	
	log_writer.write("                ***** End training *****\n")
	log_writer.close()
