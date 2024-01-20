from ctransformers import AutoModelForCausalLM



model_id = "TheBloke/Llama-2-13B-chat-GGML"
config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.1, 'stream': True}

llm = AutoModelForCausalLM.from_pretrained(model_id,
                                           model_type="llama",

                                           gpu_layers=130,

                                           **config
                                           )




prompt = '''Please parse this model in accordace with these features":

Period: H1,
Year:2015,
.... 
China Eastern Airlines Corporation Limited,2015-07-15,"China Eastern Airlines Corp. Ltd. provided earnings guidance for the first half ended June 30, 2015. The company expects net profit attributable to the equity holders of the company for the first half of 2015 to be RMB 3.5 billion to RMB 3.7 billion, representing an increase of 24900% to 26329% as compared to the corresponding period last year. Reason for the increase in profit for the current period are: During the first half of 2015, benefiting from the factors such as continual low international crude oil prices, the adjustment of China's economic structure and consumption upgrade of residents, there was strong demand for the aviation market; the Company has further enhanced its capabilities in corporate operation and management, such as operation coordination, analysis of the market and cost control to better seize market opportunities; and the company has steadily pushed forward various reforms and transformations, thereby increasing its competitiveness persistently.",1.0,The company expects net profit attributable to the equity holders of the company for the first half of 2015 to be RMB 3.5 billion to RMB 3.7 billion,H1,2015,,Net profit attributable to equity holders

        
'''

        #if correct ---> next example
        #else pick randomly from example dataset and generate stacked chain of thought
tokens = llm.tokenize(prompt)
gen = llm(prompt,stream=False)

print(gen)
