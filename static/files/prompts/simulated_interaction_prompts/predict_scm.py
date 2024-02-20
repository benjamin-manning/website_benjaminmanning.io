import openai
import replicate
import os
import sys
import re
import json
from dotenv import load_dotenv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
models = ["gpt-3.5-turbo-16k","gpt-4"]
prompt_option = ['wo_regression', 'w_regression']


def dict_to_string(d):
    s = []
    for key, value in d.items():
        s.append(f"{key}: {value} ;")
    return "".join(s)

def list_to_string(lst):
    # Check if lst is None and return an empty string
    if lst is None:
        return ""
    strings = []
    for d in lst:
        if len(lst) == 1:
            strings.append(f"{d}")
        else:
            strings.append(f"{d} + ")
    return ''.join(strings)

# Function to create the prompt
def create_prompt(datapoint):
    return f'''{dict_to_string(datapoint)}\n 
        You are being asked a question that requires a numerical response 
        in the form of an integer or decimal (e.g., -12, 0, 1, 2, 3.45, ...).
        
        Your response must be in the following format:
        {{ "explanation": "calculation for your prediction", "prediction": "<your numerical answer here>" }}

        You must only include an integer or decimal in the quoted "prediction" part of your response. 

        Here is an example of a valid response:
        {{ "explanation": "This is my calculation process...", "prediction": "100" }}'''

def call_openai(prompt, model):
    response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo-16k",
    model=model,
    messages=[
        {
        "role": "system",
        "content": "You are a social scientist"
        },
        {
        "role": "user",
        "content": prompt
        },
    ],
    temperature=0,
    max_tokens=400,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]


# Iterate over each row and get predictions
def process_row(args):
    index, row, prompt_dict, EXO_key, ENDO_key = args
    # Constructing the new dictionary 'datapoint'
    datapoint = {key: row[key] for key in EXO_key}
    print(datapoint)
    prompt = create_prompt(datapoint)
    prompt = prompt_dict + prompt
    prediction = call_openai(prompt, models[1])
    
    
    try:
        # Use regular expressions to find the number following "prediction"
        prediction_value = json.loads(prediction)["prediction"]
        print( prediction_value)
        return (index, prediction_value, True)

    except Exception as e:
        print(f"Error processing the prediction: {e}")
        return (index, prediction, False)


if __name__ == '__main__':

    filepath = '/Users/wonderland/Dropbox/robot_scientist/New_data/auction_art_3vars/3 bidders participating in an auction for a piece of art starting at fifty dollars.json'
    
    with open(filepath, 'r') as file:
        scm = json.load(file)
    scm_dict = json.loads(scm)

    Scenario = scm_dict["args"]["scenario_description"]
    Agents = scm_dict["args"]["agents_in_scenario"]
    print(Scenario)
    
    new_dict = {}
    ENDOGENOUS_VARIABLES =[]
    EXOGENOUS_VARIABLES =[]
    Level = {}
    if 'args' in scm_dict and 'variable_dict' in scm_dict['args']:
        variable_dict = scm_dict['args']['variable_dict']
        
        # go over variable_dict
        for key, value in variable_dict.items():
            if 'args' in value and 'agent_measure_question_dict' in value['args']:
                new_dict[key] = value['args']['agent_measure_question_dict']
            if value['class'] == 'EndogenousVariable':
                ENDOGENOUS_VARIABLES.append(key)
            elif value['class'] == 'ExogenousVariable':
                EXOGENOUS_VARIABLES.append(key)
            ## construct the level dict
            Level[f"{key}"] =  {"units": value['args']['units'], 'levels': value['args']['levels']}
                   
    measurementsInfo = new_dict
    print('measurementsInfo',measurementsInfo, ENDOGENOUS_VARIABLES, Level)
    
    # Read the specified columns from the CSV file
    try:
        data = pd.read_csv('data.csv')
        column_name = data.columns.tolist()
    except FileNotFoundError as e:
        error_message = str(e)
        data = None
        
    ## Get the mapping of the column names
    with open('final_mapping.json', 'r') as file2:
        mapping_dict = json.load(file2)
    # mapping_dict = json.loads(mapping)
    
    # Mapping the values
    ENDO_key = [mapping_dict[var] for var in ENDOGENOUS_VARIABLES]
    EXO_key = [mapping_dict[var] for var in EXOGENOUS_VARIABLES]
    
    accuracy = 0
    predictions = []

    # regression_result = "```\nRegressions:\n                                       Estimate    Std.Err  z-value  P(>|z|)\n final_art_price ~   \n bid1_max_budgt      0.351    0.017   21.251    0.000 \nbid2_max_budg       0.293    0.017   17.612    0.000 \nbid3_max_budg       0.312    0.017   18.924    0.000\n Intercepts:\nEstimate    Std.Err  z-value  P(>|z|)\n .final_art_pric    -5.577    6.023   -0.926    0.354```"
    
    regression_result = "```\nRegressions:\n                                       Estimate    Std.Err  z-value  P(>|z|)\n final_art_price ~   \n bid1_max_budgt       0.366    0.014   25.686    0.000 \nbid2_max_budg      0.274    0.013   20.423    0.000 \nbid3_max_budg       0.301    0.013   22.579    0.000\n Intercepts:\nEstimate    Std.Err  z-value  P(>|z|)\n .final_art_pric      -0.810    4.690   -0.173    0.863```"

    prompt_dict = {}
    prompt_dict['wo_regression'] =f'''I have just run an experiment to test the SCM. We ran the experiment onmultiple instances of GPT-4 to estimate the causal pathways outlined in the SCMs. We induced variation in the parent variables to test and get our data to estimate. With the Regression report diagram, there is also comprehensive information about the variables and the individual agents involved in the scenario. In each simulation run, each of the agents was only provided information on the variables relevant\nto themselves, a description of the scenario, their name, role, and\na goal and constraint that were relevant to the scenario that you\ndetermined for them by asking yourself ‘‘what are a reasonable goal\nand constraint for this scenario.’’ You can see for each variable the\nvalues that were varied in different runs of the experiment as the\n‘‘varied attribute levels’’ in the table. \nYour task is to predict what you believe the results for the new datapoints based on your experience. \nIn the {Scenario} scenario, we have \"agents_in_scenario\":{Agents}.\n The variables and levels are {Level} \nYour task is to predict what you believe the results for the new datapoints based on your experience given fixed values for {list_to_string(EXO_key)}.\n Predicted "{list_to_string(ENDO_key)}" for the new datapoints with a number'''

    prompt_dict['w_regression'] = f'''I have just run an experiment to test the SCM. We ran the experiment onmultiple instances of GPT-4 to estimate the causal pathways outlined in the SCMs. We induced variation in the parent variables to test and get our data to estimate. With the Regression report diagram, there is also comprehensive information about the variables and the individual agents involved in the scenario. In each simulation run, each of the agents was only provided information on the variables relevant\nto themselves, a description of the scenario, their name, role, and\na goal and constraint that were relevant to the scenario that you\ndetermined for them by asking yourself ‘‘what are a reasonable goal\nand constraint for this scenario.’’ You can see for each variable the\nvalues that were varied in different runs of the experiment as the\n‘‘varied attribute levels’’ in the table.\nIn the {Scenario} scenario, we have \"agents_in_scenario\":{Agents}.\n The variables and levels are {Level}\nYour task is to predict what you believe the results for the new datapoints based on the fitted Structural equations given fixed values for {list_to_string(EXO_key)}, and your experience in the scenario\nThe regression results for the model '{list_to_string(ENDO_key)} ~ {list_to_string(EXO_key)} ' are \n {regression_result}. \n Predicted "{list_to_string(ENDO_key)}" for the new datapoints with a number'''

    # for index, row in data.iterrows():
    #     # Constructing the new dictionary 'datapoint'
    #     datapoint = {key: row[key] for key in EXO_key}
    #     prompt = create_prompt(datapoint)
    #     prompt = prompt_dict[prompt_option[1]] + prompt
    #     # print(prompt)
    #     prediction = call_openai(prompt, models[1])
    #     print(prediction)
    #     try:
    #         # Use regular expressions to find the number following "prediction"
    #         match = re.search(r'"prediction":\s*"([\d,]+)"', prediction)
    #         if match:
    #             prediction_value = int(match.group(1).replace(',', ''))
    #             predictions.append(prediction_value)  # Append prediction to the list
    #             if prediction_value == row[ENDO_key[0]]:
    #                 accuracy += 1
    #             print(datapoint,'true value', row[ENDO_key[0]],"prediction", prediction_value)

    #             # ... existing code to compare prediction with actual value ...
    #         else:
    #             print("Prediction format is incorrect or not found.")
    #             # sys.exit()
    #             # predictions.append(prediction)
    #             predictions.append(row,prediction)

    #     except Exception as e:
    #         print(f"Error processing the prediction: {e}")
    #         sys.exit()
    
    
    args = [(index, row, prompt_dict[prompt_option[1]], EXO_key, ENDO_key) for index, row in data.iterrows()]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_row, args))

    for index, prediction_value, is_correct in results:
        if prediction_value is not None:
            predictions.append(prediction_value)
            if is_correct:
                accuracy += 1

    print(predictions)
    # Add predictions as a new column
    data['prediction'] = predictions
        # sys.exit()
    
    # Add predictions as a new column
    # data['prediction'] = predictions
                
    # Calculate the accuracy rate
    accuracy_rate = accuracy / len(data)
    print(f"Accuracy rate: {accuracy_rate}")

    # Save the DataFrame with the new column to a new CSV file
    data.to_csv(f'finished_using_prediction_gpt4_{prompt_option[1]}.csv', index=False)