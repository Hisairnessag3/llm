from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-LLM-13B")
tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-LLM-13B", use_fast=False)

# Put your input here:
user_input = '''Please parse this model in accordace with these features":

Period: H1,
Year:2015,
.... 
China Eastern Airlines Corporation Limited,2015-07-15,"China Eastern Airlines Corp. Ltd. provided earnings guidance for the first half ended June 30, 2015. The company expects net profit attributable to the equity holders of the company for the first half of 2015 to be RMB 3.5 billion to RMB 3.7 billion, representing an increase of 24900% to 26329% as compared to the corresponding period last year. Reason for the increase in profit for the current period are: During the first half of 2015, benefiting from the factors such as continual low international crude oil prices, the adjustment of China's economic structure and consumption upgrade of residents, there was strong demand for the aviation market; the Company has further enhanced its capabilities in corporate operation and management, such as operation coordination, analysis of the market and cost control to better seize market opportunities; and the company has steadily pushed forward various reforms and transformations, thereby increasing its competitiveness persistently.",1.0,The company expects net profit attributable to the equity holders of the company for the first half of 2015 to be RMB 3.5 billion to RMB 3.7 billion,H1,2015,,Net profit attributable to equity holders

'''
# Simply use your input as the prompt for base models
prompt = user_input

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
outputs = model.generate(input_ids=inputs, max_length=2048)[0]

answer_start = int(inputs.shape[-1])
pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)

print(f'### User Input:\n{user_input}\n\n### Assistant Output:\n{pred}')