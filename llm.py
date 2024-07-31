from typing import Optional, List, Dict
import re
import google.generativeai as palm
import openai



class BrainLLM:

    def __init__(self, model:str,
                temperature: float, 
                llm_key:str, 
                initial_prompt_path: str, 
                feedback_prompt_path: str) -> None:
        self.model = model
        self.temp = temperature
        self.key = llm_key
        with open(initial_prompt_path, 'r') as file:
            self.initial_prompt = file.read()
        with open(feedback_prompt_path, 'r') as file:
            self.feedback_prompt = file.read()


    def __call__(self, task:str, object_list: str, env_information:str,
                 feedback_mode: bool, feedback: Optional[str] = None,
                 feedback_step: Optional[int] = 7, original_plan: Optional[str] = None) -> None:
        if feedback_mode:
            prompt = self.feedback_prompt.format(input=task, 
                                                 object_list=object_list, 
                                                 feedback_message=feedback,
                                                 feedback_step=feedback_step,
                                                 init_plan=original_plan,
                                                 env_information=env_information)
        else:
            prompt = self.initial_prompt.format(input=task, object_list=object_list,
                                                env_information=env_information)

        if(self.model=='models/text-bison-001'):
            palm.configure(api_key=self.key)
            completion = palm.generate_text(model=self.model,
                                            prompt=prompt,
                                            temperature=self.temp,
                                            max_output_tokens=800)
            llm_output =  completion.result
        elif(self.model=='gpt-3.5-turbo' or self.model=='gpt-4'):
            client = openai.OpenAI(api_key=self.key)
            completion = client.chat.completions.create(model=self.model,
                                                        temperature=self.temp,
                                                        messages=[{"role": "user", "content": prompt}],
                                                        max_tokens=256)
            llm_output = completion.choices[0].message.content
        else:
            client = openai.OpenAI(api_key=self.key)
            response = client.completions.create(model=self.model,
                                                 prompt=prompt,
                                                 temperature=self.temp,
                                                 max_tokens=256)
            llm_output = response.choices[0].text

        # Check if it starts with 'Explanation' or 'explanation'
        if llm_output.lower().startswith('explanation'):
        # Remove the first line
            _, _, llm_output = llm_output.partition('\n')

        if not feedback_mode:
            print(f'First plan is:\n{llm_output}')
        else:
            print(f'Plan after feedback, feedback step={feedback_step}, feedback:{feedback}\nPlan before this step: {original_plan}\nNew plan:\n{llm_output}')
        print('#####Brain LLM End#####')
        # First element of the queue is the initial plan
        # We could have created another queue for this process 
        # But for now, there is no need for this.
        output = [substring.strip() for substring in re.split(r'\d+: ', llm_output) if substring and substring.strip()]
        return output
        
        

class BodyLLM:
    def __init__(self, model:str, temperature: float, llm_key:str, prompt_path: str, cube_positions: Dict,
                 locations: Dict, controller) -> None:
        self.model = model
        self.temp = temperature
        self.key = llm_key
        self.cube_positions = cube_positions
        self.locations = locations
        self.controller = controller
        with open(prompt_path, 'r') as file:
            self.prompt = file.read()
    
    def __call__(self, task: str, object_list: List) -> str:
        prompt = self.prompt.format(input=task, object_list=object_list)
        if(self.model=="models/text-bison-001"):
            palm.configure(api_key=self.key)
            completion = palm.generate_text(model=self.model,prompt=prompt,temperature=self.temp,max_output_tokens=800)
            llm_output =  completion.result
            # rarely palm api does not give an output, this line of code is to fix this problem
            if llm_output is None: 
                llm_output =  '[]'
        elif(self.model=="gpt-3.5-turbo" or self.model=="gpt-4"):
            client = openai.OpenAI(api_key=self.key)
            completion = client.chat.completions.create(model=self.model,
                                                        temperature=self.temp,
                                                        messages=[{"role": "user", "content": prompt}],
                                                        max_tokens=25)
            llm_output = completion.choices[0].message.content
        else:
            client = openai.OpenAI(api_key=self.key)
            response = client.completions.create(model=self.model,
                                                 prompt=prompt,
                                                 temperature=self.temp,
                                                 max_tokens=25)
            llm_output = response.choices[0].text.strip()
        
        # Although it is extremely rare, sometimes body llm halucinates and give more than one line in its output.
        # This piece of code takes the first line as an output in this type of incidents.
        
        for line in llm_output.splitlines():
            if line.strip():  # Check if the line is not empty after stripping whitespace
                # output = self.parse_body_llm(instruction= line)
                try:
                    output = self.parse_body_llm(instruction= line)
                    # print(line)
                except Exception as e:
                    output = str(e)
                print('###Body LLM Start###')
                print(f'Body llm input:{task}')
                print(f'Body llm output:{llm_output}')
                print(f'Body llm output after parser:\n{output}')
                print('###Body LLM End###')
                return output
            

        
    
    def parse_body_llm(self, instruction:str):
        # Split the instruction into components
        components = instruction.split()
        # Initialize variables for action, object, and location
        action = None
        object_ = None
        location = None

        # Check each component for the action, object, and location
        for component in components:
            if '[grasp]' in component:
                action = 'grasp'
            elif '[put]' in component:
                action = 'put'
            elif '[return_base]' in component:
                action = 'return_base'
            elif '[pass]' in component:
                action = 'pass'
            elif component.startswith('<') and component.endswith('>') and component != '<rob0>':
                if object_ is None:
                    object_ = component
                else:
                    location = component
        # Return the results based on the identified action
        if action == 'grasp':
            cube_position = self.cube_positions[object_]
            # print(cube_position)
            if (abs(cube_position[0]) + abs(cube_position[1])) > 2.0:
               raise Exception(f'{object_} is out of the workspace of the robot.')
            if (abs(cube_position[0]) + abs(cube_position[1])) < 0.2:
               raise Exception(f'{object_} is out of the workspace of the robot.')
            return self.controller('grasp', self.cube_positions[object_])
        elif action == 'put':
            loc = self.locations[location]
            if (abs(loc[0]) + abs(loc[1])) > 1.0:
                raise Exception(f'{location} is out of the workspace of the robot.')
            return self.controller('place', self.locations[location])
        elif action == 'return_base':
            return self.controller('return', [0.3, 0.0, 0.5])
        elif action == 'pass':
            return None