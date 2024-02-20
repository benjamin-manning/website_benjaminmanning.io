import sys
import os
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.append('../LLM')
sys.path.append('../JudeaPearl')
sys.path.append('../Question')

from LLM import LanguageModel, LLMMixin, llm_json_loader
from AgentBuilder import AgentBuilder
from Human import Human
from Interaction import SocialInteraction
from StructuralCausalModelBuilder import StructuralCausalModelBuilder
from Prompting import PromptMixin

from jinja2 import Environment, FileSystemLoader
templates_dir = '/Users/wonderland/Desktop/AgentHub/robot_scientist/JudeaPearl/prompt_templates'
env = Environment(loader=FileSystemLoader(templates_dir))


def build_agent(scm_json):
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=.4, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm_json)
    
    agent_builder = AgentBuilder(template_dir=templates_dir)
    agent_builder.add_LLM(LLM)
    agent_builder.add_scm(loaded_scm)
    
    exp_role = agent_builder.backend_build_agents()
    
    varied_attributes_dict =  agent_builder.backend_return_varied_attributes()
    
    return exp_role, varied_attributes_dict
    
    
def build_interaction(scm_json):
        
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=.3, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm_json)
    agent_builder = AgentBuilder(template_dir=templates_dir)
    agent_builder.add_LLM(LLM)
    agent_builder.add_scm(loaded_scm)
    
    gen_func_type, order_dict = agent_builder.backend_get_interaction_info()
    
    return gen_func_type, order_dict


def call_openai(agent_list, order_dict,scenario, interaction_type, max_interactions, OPERATIONALIZATION, ENDOGENOUS_VARIABLES):
    agentsInfo = agent_list
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=.4)
    L_S = LanguageModel(family="openai", model="gpt-4", temperature=.7)
    
    agents = {}
    agent_list = []
    conversation_history = []
    conversation_history_simple = []

    for agent_type, attributes in agentsInfo.items():
        agent = Human(attributes)
        agent.add_LLM(LLM)
        agents[agent_type] = agent
        agent_list.append(agent)
        
    if interaction_type == 'center random':
        center = order_dict['central agent']
        if order_dict['order'] is None:
            for agent_type, attributes in agentsInfo.items():
                if agent_type != center:
                    order_dict['order'].append(agent_type)

    # print(order_dict)
    S = SocialInteraction(agent_list, scenario=scenario)
    S.add_LLM(L_S)
    generator = S.gen_func_dispatch[interaction_type](order_dict)
    FirstAgent = agents[next(generator)]
    SecondAgent = agents[next(generator)]
    others = [agent for agent in agent_list if agent != FirstAgent]
    statement = FirstAgent.make_public_statement(others, scenario, conversation_history_simple)
    print("No.0", statement)
        

    interactions = 0
    conversation_history.append({FirstAgent.name: statement})
    conversation_history_simple.append({FirstAgent.name: statement['statement']})
    if statement['is_rational'] is False:
        print("Error! The response is not rational")
        return conversation_history

    while True:
        interactions += 1
        others = [agent for agent in agent_list if agent != SecondAgent]
        newstatement = SecondAgent.make_public_statement(others, scenario, conversation_history_simple)
        name = SecondAgent.name
        FirstAgent = SecondAgent
        SecondAgent = agents[next(generator)]
        statement = newstatement
        print(f"No.{interactions}", statement)
        
        conversation_history.append({name: newstatement})
        conversation_history_simple.append({name: statement['statement']})
        if statement['is_rational'] is False:
            print("Error! The response is not rational")
            break

        to_continue = FirstAgent.to_continue_or_to_finish(scenario, agent_list, OPERATIONALIZATION= OPERATIONALIZATION,ENDOGENOUS_VARIABLES=ENDOGENOUS_VARIABLES,history=conversation_history_simple)
        if not to_continue:
            print('>>>>>><<<<<<', to_continue)
            break
        if interactions > max_interactions:
            break  
        
    return [conversation_history, conversation_history_simple]


def call_measurement(history, measurementsInfo, agent_str, ENDOGENOUS_VARIABLES, SCNEARIO_DESCRIPTION, OPERATIONALIZATION):
    
    responses = {}
     
    ## Set the temperature to be lower 
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=.4)

    agents = {}
    agent_list = []
    
    replacement_dict = {"your role is": "role", "your name": "name"}
    NAME_AND_ROLE = [f"{replacement_dict[key]}: {value}" for key, value in agent_str.items() if key in ["your role is", "your name"]]
    EXDOGENOUS = [variable_name for variable_name in measurementsInfo if variable_name not in ENDOGENOUS_VARIABLES]

    for agent_type, attributes in agent_str.items():
        agents[agent_type] = Human(attributes)
        agents[agent_type].add_LLM(LLM)
        agent_list.append(agents[agent_type])
            
    responses = {}

    for variable_name in measurementsInfo:
        if variable_name in ENDOGENOUS_VARIABLES:
            for agent_name, questions_data in measurementsInfo[variable_name].items():
                
                questions = questions_data if isinstance(questions_data, list) else [questions_data]
                
                for question in questions:
                    if variable_name not in responses:
                        responses[variable_name] = {}
                    if agent_name not in responses[variable_name]:
                        responses[variable_name][agent_name] = {}
                        
                    # Oracle case
                    if agent_name == 'oracle':
                        prompt1 = PromptMixin().generate_prompt("ask_oracle_prescriptively.txt", template_dir = self.template_dir, **prompt_params)
                        
                        f"""
                        We have just completed a simulation of the following scenario: {SCNEARIO_DESCRIPTION}, with these human agents: {NAME_AND_ROLE}. Here is the transcript of their conversation: {history}. Your task is to answer the following question: '{question}'.
                        When answering the question, please keep the following things in mid:
                        1. You should base you answer frist on the agent's personal characteristics provided and the conversation history.
                        2. This simulated conversation was run as an experiment to test the effects of changing different attributes on the {EXDOGENOUS}. Your answer to the question will be directly used to operaionalize the measurement of the {variable_name} for data data analysis, which we originally chose to operationalize like this: {OPERATIONALIZATION}.
                        You should try as hard as possibly to accurately answer the question within the context of the agents characteristics, conversation, and usage for analyzing the simulation, but if you truly cannot answer the questions, you can say that you don't know.Format your response as a json in this form and make sure that all keys and items are in double quotes correctly: {{"explanation": "short explanation for choice”, "answer": "your answer to the question do get the data for the analysis."}}"""
                        
                        survey_answer1 = LLM.call_llm(prompt1)
                        # prompt2 = prompt1 + f'\[oracle: {survey_answer1}\]' + '\[ user: Think about your answer again. Based on this new perspective, make a second estimate. Using the format {{"answer": "your answer to the question","explanation": "short explanation for choice"}}\]'
                        # # store to response
                        # survey_answer2 = LLM.call_llm(prompt2)
                        # prompt3 = prompt2 + f'\[oracle: {survey_answer2}\]' + '\[ user: Think about your answer again. Based on this new perspective, make a third estimate. Using the format {{"answer": "your answer to the question","explanation": "short explanation for choice"}}\]'
                        # survey_answer3 = LLM.call_llm(prompt3)
                        responses[variable_name][agent_name][question] = survey_answer1 
                        #[survey_answer1,survey_answer2,survey_answer3]

                    # Agent cases
                    else:
                        others = [agent for agent in agent_list if agent != agent_name]
                        survey_answer = agents[agent_name].survey(others, SCNEARIO_DESCRIPTION, question, history, EXDOGENOUS=EXDOGENOUS, VARIABLE=variable_name, OPERATIONALIZATION=OPERATIONALIZATION)
                        # store to response
                        responses[variable_name][agent_name][question] = survey_answer

    
    # for key in measurementsInfo:
    #     for subkey, subvalue in measurementsInfo[key].items():
    #         # print("<<<", key, subkey, subvalue)
    #         questions = subvalue if isinstance(subvalue, list) else [subvalue]
            
    #         # if key not in responses:
    #         for question in questions:
    #             if question not in responses:
    #                 responses[question] = {}
    #             if subkey not in responses[question]:
    #                 responses[question][subkey] = {}

    #         # Oracle case
    #         if subkey == 'oracle':
    #             context = f"""
    #                 You are the oracle.
    #                 You have recently watched a social interaction. {history}
    #                 """   
    #             for question in questions:    
    #                 prompt = f"""{context}
    #                     How do you respond to the following question: {question}
    #                     Your response:"""
    #                 survey_answer = LLM.call_llm(prompt)
    #                 # store to response
    #                 responses[question][subkey]=survey_answer

    #         # Agent cases
    #         else:
    #             for question in questions:   
    #                 survey_answer = agents[subkey].survey(question, history)
    #                 # store to response
    #                 # responses[key] = {subkey: [survey_answer]}
    #                 responses[question][subkey]=survey_answer
    return responses

def survey_wrapper(args):
    history, agent_str, measurementsInfo, ENDOGENOUS_VARIABLES,scenario, OPERATIONALIZATION = args
    return call_measurement(history=history, agent_str=agent_str, measurementsInfo=measurementsInfo, ENDOGENOUS_VARIABLES= ENDOGENOUS_VARIABLES, SCNEARIO_DESCRIPTION=scenario, OPERATIONALIZATION= OPERATIONALIZATION)

def generate_all_combinations_with_mapping(agentsInfo, variations):
    combined_dicts = []
    
    # Extract all the keys and the corresponding variation lists
    variation_keys = list(variations.keys())
    variation_list = [variations.get(var, {}) for var in variation_keys]
    
    # To store the attribute value mapping
    attribute_value_mapping = {}
    
    # Get the maximum length for each variation
    max_len_list = [max([len(values) for attribute_data in var.values() for attribute, values in attribute_data.items()]) for var in variation_list]
    
    # Use product to iterate over all possible combinations
    from itertools import product
    for idx, combination in enumerate(product(*[range(max_len) for max_len in max_len_list])):
        combined_dict = {}
        variation_dict = {}
        
        for role in agentsInfo:
            agent = agentsInfo[role].copy()
            
            # For each variation, update the attribute
            for var_index, variation_data in enumerate(variation_list):
                for attribute, attribute_values in variation_data.get(role, {}).items():
                    if combination[var_index] < len(attribute_values):
                        agent[attribute] = attribute_values[combination[var_index]]
                        variation_dict[variation_keys[var_index]] = attribute_values[combination[var_index]]
            
            combined_dict[role] = agent
        
        # Update the attribute value mapping
        attribute_value_mapping[str(idx)] = variation_dict
        combined_dicts.append(combined_dict)

    return combined_dicts, attribute_value_mapping


def get_simple_history(histories):
    for history in histories:
        for record in history:
            for name, info in record.items():
                if "is_rational" in info:
                    del info["is_rational"]
    return histories



if __name__ == '__main__':
    # agent_lists = ["less than 1 year", "1-2 years", "2-5 years"]
    # proprty_name = {"seller": {"your experience in selling": agent_lists}}
    # all_agent_lists = []
    # filepath = '/Users/wonderland/Desktop/test/a_family_income.json'
    # filepath = '/Users/wonderland/Dropbox/robot_scientist/SCM/.json'
    filepath = '/Users/wonderland/Dropbox/robot_scientist/SCM/waitier_other_applicants.json'
    
    with open(filepath, 'r') as file:
        scm = json.load(file)
    scm_dict = json.loads(scm)
    print(type(scm),type(scm_dict))

    scenario = scm_dict["args"]["scenario_description"]
    max_interactions = 70
    print(scenario)
    
    ## get the measurement Info
    new_dict = {}
    ENDOGENOUS_VARIABLES =[]
    OPERATIONALIZATION = []
    if 'args' in scm_dict and 'variable_dict' in scm_dict['args']:
        variable_dict = scm_dict['args']['variable_dict']
        
        # go over variable_dict
        for key, value in variable_dict.items():
            if 'args' in value and 'agent_measure_question_dict' in value['args']:
                new_dict[key] = value['args']['agent_measure_question_dict']
            if value['class'] == 'EndogenousVariable':
                ENDOGENOUS_VARIABLES.append(key)
                OPERATIONALIZATION.append(value['args']['operationalization_dict']['operationalization'])
                   
    measurementsInfo = new_dict
    print('measurementsInfo',measurementsInfo, ENDOGENOUS_VARIABLES, OPERATIONALIZATION)
    

    ## Added the check point
    agent_filepath = f"agent_{scenario}.json"
    ## give the scm (give by uploading)
    if os.path.exists(agent_filepath):
        print('agents info already built')
        # If the file exists, read the content
        with open(agent_filepath, "r") as file:
            data_dict = json.load(file)
            # print(type(data_dict))
        exp_role = data_dict["exp_role"]
        variations = data_dict["variations"]
        interaction_type = data_dict["interaction_type"]
        order_dict = data_dict["order_dict"]
        measurementsInfo = data_dict["measurementsInfo"]
        combined_dicts, attribute_value_mapping = generate_all_combinations_with_mapping(exp_role, variations)
        print('variation',len(combined_dicts),combined_dicts)
        # sys.exit()
    else:
        exp_role, variations = build_agent(scm)
        interaction_type, order_dict = build_interaction(scm)
        print(exp_role, variations)
        print('Interacton', interaction_type, order_dict)
        ## add a check point
        combined_dicts, attribute_value_mapping = generate_all_combinations_with_mapping(exp_role, variations)
        
        data_to_save = {
            "exp_role": exp_role,
            "variations": variations,
            "interaction_type": interaction_type,
            "order_dict": order_dict,
            'measurementsInfo': measurementsInfo,
            'combined_dicts': combined_dicts
        }

        # Convert the dictionary to a JSON-formatted string
        json_str = json.dumps(data_to_save, indent=4)  # indent=4 for pretty printing

        # Save the string to a file
        with open(f"agent_{scenario}.json", "w") as file:
            file.write(json_str)
    
    # print(order_dict)
    
    ## Added the check point
    history_filepath = f"history_{scenario}.json"
    if os.path.exists(history_filepath):
        print('agents history already there')
        # If the file exists, read the content
        with open(history_filepath, "r") as file:
            data_dict = json.load(file)
            histories = data_dict['histories']
            histories_simple = get_simple_history(histories)
        print('variations', len(histories))    
    else:
        # use functools.partial to preset the parameters
        partial_call = partial(call_openai, scenario=scenario, order_dict = order_dict ,interaction_type=interaction_type, max_interactions=max_interactions, OPERATIONALIZATION= OPERATIONALIZATION, ENDOGENOUS_VARIABLES= ENDOGENOUS_VARIABLES)

        with ProcessPoolExecutor() as executor:
            # results = list(executor.map(partial_call, combined_dicts))
            combined_results = list(executor.map(partial_call, combined_dicts))

        # separate the conpound result and simple
        histories = [res[0] for res in combined_results if len(res) > 0]
        histories_simple = [res[1] for res in combined_results if len(res) > 1]
            
            
        data_to_save = {
            "histories": histories
        }
        # Convert the dictionary to a JSON-formatted string
        json_str = json.dumps(data_to_save, indent=4)  # indent=4 for pretty printing

        # Save the string to a file
        with open(f"history_{scenario}.json", "w") as file:
            file.write(json_str)
               
    # for result in results:
    #     print("_____________")
    #     print(result)
        
    ## survey the agents
    # partial_survey = partial(call_measurement, measurementsInfo = measurementsInfo)

    # use lambda func to receive 2 para：history and agent_str


    # with ProcessPoolExecutor() as executor: 
        # surveys = list(executor.map(survey_func, zip(histories, combined_dicts)))
    
    with ProcessPoolExecutor() as executor:
        args = zip(histories_simple, combined_dicts, [measurementsInfo]*len(histories_simple), [ENDOGENOUS_VARIABLES]*len(histories_simple), [scenario]*len(histories_simple),[OPERATIONALIZATION]*len(histories_simple))
        surveys = list(executor.map(survey_wrapper, args))
    
    # with ProcessPoolExecutor() as executor:
        # surveys = list(executor.map(partial_survey, histories))
        
    data_to_save_all = {
            "scm": scm,
            "agents": combined_dicts,
            "interaction": histories,
            "survey": surveys
        }
    
    json_str1 = json.dumps(data_to_save_all, indent=4)  # indent=4 for pretty printing

    # Save the string to a file
    with open(f"raw_result_{scenario}.json", "w") as file:
        file.write(json_str1)
    
    ## Old attribute list
    # attribute_values_order = variations

    # # Initialize the new data structure
    # reordered_data = {}
    
    # # For each attribute value, reorder the data
    # for i, attribute_value in enumerate(attribute_values_order, 1):
    #     reordered_data[str(i)] = {}
    #     # Add the matching agents, interaction, and survey entries
    #     for key in ['agents', 'interaction', 'survey']:
    #         # if i <= len(data_to_save_all[key]):
    #         #     reordered_data[str(i)].append({key: data_to_save_all[key][i-1]})
    #         if i <= len(data_to_save_all[key]):
    #             reordered_data[str(i)][key] = data_to_save_all[key][i-1]
                
    # # attribute_value_mapping = {str(i): {"your annual income": value} for i, value in enumerate(attribute_values_order, 1)}
    # # Filtering out empty subdicts
    # non_empty_subdicts = {k: v for k, v in attribute_values_order.items() if v}
    # attribute_value_mapping = {}
    # counter = 1
    # for key, subdict in non_empty_subdicts.items():
    #     for subkey, values in subdict.items():
    #         for value in values:
    #             attribute_value_mapping[str(counter)] = {subkey: value}
    #             counter += 1

    # # print(attribute_value_mapping)
    
    ## New
    # extracted_variations = {}

    # for key, value in variations.items():
    #     non_empty_list = None
    #     for role, agent_data in value.items():
    #         for agent, list_values in agent_data.items():
    #             if list_values and not non_empty_list:  # check if the list is non-empty and we haven't found a non-empty list yet
    #                 non_empty_list = list_values
    #                 break
    #     if non_empty_list:
    #         extracted_variations[key] = non_empty_list
            
    # all_combinations_list = generate_combinations(extracted_variations)

    # Convert the list of combinations to the desired format
    # attribute_value_mapping = {str(i): combo for i, combo in enumerate(all_combinations_list)}
    
    reordered_data = {}
    # for i, combo in enumerate(all_combinations_list):
    for key in attribute_value_mapping.keys():
        # reordered_data[str(i)] = {}
        # Add the matching agents, interaction, and survey entries
        reordered_data[key] = {}
        # for key in ['agents', 'interaction', 'survey']:
        #     if i <= len(data_to_save_all[key]):
        #         reordered_data[str(i)][key] = data_to_save_all[key][i-1]
        for subkey in ['agents', 'interaction', 'survey']:
            if int(key) < len(data_to_save_all[subkey]):
                reordered_data[key][subkey] = data_to_save_all[subkey][int(key)]
    
    combined_data = {
        "scm": scm,
        "data": reordered_data,
        "attribute_value_mapping": attribute_value_mapping
    }

    # Convert the dictionary to a JSON-formatted string
    json_str = json.dumps(combined_data, indent=4)  # indent=4 for pretty printing

    # Save the string to a file
    with open(f"result_{scenario}.json", "w") as file:
        file.write(json_str)

