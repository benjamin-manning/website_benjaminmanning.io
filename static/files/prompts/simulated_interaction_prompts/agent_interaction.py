import sys
import time
import inspect 
import functools
import json

from collections import namedtuple

sys.path.append('../LLM')
sys.path.append('../Serialization')

from LLM import LanguageModel
from Serialize import RegisteredSerializable
from Prompting import PromptMixin


def dict_to_string(d):
    s = []
    for key, value in d.items():
        # remove 'name' and 'role'
        if key not in ['your name', 'your role is']:
            s.append(f"{key}: {value}")
    return "".join(s)

def list_to_string(lst):
    # Check if lst is None and return an empty string
    if lst is None:
        return ""
    
    strings = []
    for d in lst:
        for key, value in d.items():
            strings.append(f"{key}: {value}")
    return ''.join(strings)


MemoryInput = namedtuple("MemoryInput", "args kwargs time")
MemoryRecord = namedtuple("MemoryRecord", "function inputs output time")

def is_yes(str):
    """Returns True if the string is some form of a yes""" 
    return "yes" in str.lower() in str.lower()

class MemoryLocation(RegisteredSerializable):
    """This will let us expand memory types more easily in the future"""
    def __init__(self):
        self.complete = []
        self.simple = []

    def __repr__(self):
        return f"complete: {self.complete}\nsimple: {self.simple}"

def remember(*memory_types):
    """
    This decorator remembers the inputs and outputs of a function.
    
    args: me
    
    """
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            inputs = MemoryInput(args, kwargs, time.time())
            result = func(self, *args, **kwargs)
            record = MemoryRecord(func.__name__, inputs, result, time.time())
            
            for memory_type in memory_types:
                if hasattr(self.memory_locations, memory_type):
                    memory_list = getattr(self.memory_locations, memory_type)
                    memory_list.append(record)
                    setattr(self.memory_locations, memory_type, memory_list)
                else: 
                    raise ValueError("Invalid memory type. Refactor MemoryLocations class.")
                
            return result
        return wrapper
    return actual_decorator

class Human(RegisteredSerializable, PromptMixin):
    def __init__(self, attributes, template_dir: str ="prompt_templates"):
        
        self.template_dir = template_dir
        
        for key, value in attributes.items():
            if key == "LLM" and isinstance(value, dict):  # Convert dictionaries to LanguageModel
                value = LanguageModel.from_dict(value)
            setattr(self, key, value)
        self.attributes = attributes
        self.name = attributes['your name']
        #Memory of all things that an agent has said or another agent has said to it
        self.memory_locations = MemoryLocation()
        try:
            _ = self._goal  # check to see if goal is defined
        except AttributeError:
            raise Exception("You must specify a goal for the human.")
        

    def call_llm(self):
        raise NotImplementedError("This method gets implemented when you add an LLM")

    def add_LLM(self, LLM):
        self.LLM = LLM
        self.call_llm = self.LLM.call_llm

    @staticmethod
    def public_knowledge(counterparty):

        replacement_dict = {"your role is": "role", "your name": "name"}
    
        # List comprehension to generate the output
        return [f"{replacement_dict[key]}: {value}" for key, value in 
            counterparty.attributes.items() if key in ["your role is", "your name"]]

    # @remember('complete')
    def current_context(self, history = None):

        prompt_params = { 
            "role": self.attributes['your role is'],
            "name": self.attributes['your name'],
            "arrtibutes": dict_to_string(self.attributes),
            "history": list_to_string(history)
        }
        prompt = self.generate_prompt("current_context.txt", template_dir = self.template_dir, **prompt_params)
        
        return prompt 
    
    @remember('complete')
    def final_context(self, group_knowledge, scenario_description, history):
        
        prompt_params = { 
            "attributes": dict_to_string(self.attributes),
            "history": history,         
            "scenario_description": scenario_description,
            "group_knowledge": group_knowledge,
            "_goal": self._goal,
            "_constraint": self._constraint
        }
        prompt = self.generate_prompt("instantiate_agent.txt", template_dir = self.template_dir, **prompt_params)
        
        return prompt
        
    @remember('complete')
    def survey(self, counterparties, scenario_description, question, history, EXDOGENOUS, VARIABLE, OPERATIONALIZATION):
        
        group_knowledge = [self.public_knowledge(counterparty) for counterparty in counterparties]
        context = self.final_context(group_knowledge, scenario_description, history)
        
        prompt_params = { 
            "context": context,
            "question": question,
            "EXDOGENOUS": EXDOGENOUS,
            "VARIABLE": VARIABLE,
            "OPERATIONALIZATION": OPERATIONALIZATION
            
        }
        prompt = self.generate_prompt("survey_agent.txt", template_dir = self.template_dir, **prompt_params)
        
        return self.call_llm(prompt)
    
    def is_rational(self, statement, history= None):
        return True


    def make_public_statement(self, counterparties, scenario_description, history = None):
        
        group_knowledge = [self.public_knowledge(counterparty) for counterparty in counterparties]

        if not history:
            STRING = "You will be the first person to speak"
        else:
            STRING = self.current_context(history)
    
        prompt_params = { 
            "scenario_description": scenario_description,
            "group_knowledge": group_knowledge,
            "STRING": STRING
        }
        prompt = self.generate_prompt("make_statement.txt", template_dir = self.template_dir, **prompt_params)
        
        statement = self.call_llm(prompt)
        is_rational = self.is_rational(statement, history)
        return {'statement':statement, 'is_rational': is_rational}


    def to_continue_or_to_finish(self, scenario, agents,ENDOGENOUS_VARIABLES, OPERATIONALIZATION, history=None):
        
        group_knowledge = [self.public_knowledge(agent) for agent in agents]
        
        prompt_params = { 
            "scenario": scenario,
            "group_knowledge": group_knowledge,
            "ENDOGENOUS_VARIABLES": ENDOGENOUS_VARIABLES,
            "OPERATIONALIZATION": OPERATIONALIZATION,
            "history": list_to_string(history)
        }
        prompt = self.generate_prompt("to_continue_or_to_finish.txt", template_dir = self.template_dir, **prompt_params)
        
        response = self.call_llm(prompt)
        print(response)
        if "continue" in response.lower():
            return True
        else:
            return False 

    
    def how_to_you_think_other_person_will_respond(self, question):
        pass

    @remember('simple', 'complete')
    def does_this_response_help_your_goal(self, statement, response):
        
        prompt_params = { 
            "statement": statement,
            "_goal": self._goal,
            "response": response
        }
        prompt = self.generate_prompt("is_rational.txt", template_dir = self.template_dir, **prompt_params)
        
        return self.call_llm(prompt)
    
    def show_memory(self):
        return self.memory_locations
    
    def __eq__(self, other):
        if isinstance(other, Human):
            # Compare the relevant attributes here and return True if they are equal
            return self.attributes == other.attributes
        return False
