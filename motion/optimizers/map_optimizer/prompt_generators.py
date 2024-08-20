import random
import json
from typing import Dict, Any, List, Tuple

from rich.console import Console

from motion.optimizers.utils import LLMClient
from motion.optimizers.map_optimizer.utils import generate_and_validate_prompt


class PromptGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        console: Console,
        config: Dict[str, Any],
        max_threads: int,
        is_filter: bool = False,
    ):
        self.llm_client = llm_client
        self.console = console
        self.config = config
        self.max_threads = max_threads
        self.is_filter = is_filter

    def _generate_validator_prompt(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
    ) -> str:
        system_prompt = "You are an AI assistant tasked with creating custom validation prompts for data processing operations. Your goal is to create a prompt that will assess how well the operation performed its intended task."

        prompt = f"""
        Analyze the following operation and its input/output:

        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Sample Input & Output: {json.dumps(output_data[0] if output_data else {}, indent=2)}
        Task Prompt: {op_config.get('prompt', 'N/A')}

        Based on this information, create a custom validator prompt that will assess how well the original task was performed. The prompt should ask specific questions about the quality and completeness of the output, such as:
        1. Are there any instances of the target information missed?
        2. Would the output improve if the input was analyzed more carefully?
        3. Is the output format correct and consistent?
        4. Are there any errors or inconsistencies in the extracted information?

        Provide your response as a single string containing the custom validator prompt.
        """

        parameters = {
            "type": "object",
            "properties": {"validator_prompt": {"type": "string"}},
            "required": ["validator_prompt"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)["validator_prompt"]

    def _get_improved_prompt(
        self,
        op_config: Dict[str, Any],
        assessment: Dict[str, Any],
        input_data_sample: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_prompt = "You are an AI assistant tasked with improving prompts for data processing operations."

        random_sample = random.choice(input_data_sample) if input_data_sample else {}

        prompt = f"""
        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Current Prompt: {op_config.get('prompt', 'N/A')}

        Input Data Sample:
        {json.dumps(random_sample, indent=2)}

        Use the following feedback to improve the current prompt:
        {json.dumps(assessment['improvements'], indent=2)}

        Improve the current prompt to better handle the input data and produce more accurate results.     
        Note: The new prompt should only include the variables present in the current prompt verbatim. Do not introduce any new variables.

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "new_prompt": {"type": "string"},
            },
            "required": ["new_prompt"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        improved_op_config = op_config.copy()
        improved_op_config["prompt"] = result["new_prompt"]
        return [improved_op_config]

    def _get_combine_prompt(
        self,
        op_config: Dict[str, Any],
        sample_output: List[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        """
        Generate a combine prompt for merging chunk results in a map-reduce operation.

        This method creates a prompt that will be used to combine the results from
        processing individual chunks of data in a map-reduce operation. The combine
        prompt is designed to accomplish the original task by merging the outputs
        from various chunks.

        Args:
            op_config (Dict[str, Any]): The configuration of the original operation,
                including the original prompt and output schema.
            sample_output (List[Dict[str, Any]]): A list of sample outputs from
                processing various chunks. Each item in the list represents the
                output from a single chunk.

        Returns:
            Tuple[str, bool]: A tuple containing:
                - A Jinja2 template string that serves as the combine prompt.
                  This prompt will be used to merge the results from individual
                  chunks to produce the final output of the map-reduce operation.
                - A boolean indicating whether the combine operation is commutative.

        The method performs the following steps:
        1. Extracts relevant information from the op_config, including the original
           prompt and output schema.
        2. Prepares sample inputs based on the sample_output and the output schema.
        3. Constructs a base prompt that includes the original prompt, output schema,
           and sample inputs.
        4. Uses the LLM to generate a combine prompt based on the base prompt and
           specific guidelines.
        5. Validates the generated prompt to ensure it meets the required format
           and uses the correct variables.
        6. Determines whether the combine operation is commutative.

        Note:
            The generated combine prompt is constrained to use only the 'values'
            variable, which contains all chunk results. It must be a valid Jinja2
            template and avoid using complex logic or filters.

        Raises:
            Any exceptions raised by the underlying generate_and_validate_prompt
            method, which may include validation errors or LLM-related issues.
        """
        system_prompt = "You are an expert data processing assistant, decomposing a task into subtasks and joining the reults."

        # Prepare sample inputs for the combine prompt
        schema = op_config["output"]["schema"]
        schema_keys = list(schema.keys())
        if self.is_filter:
            schema_keys.append("_short_explanation")

        sample_inputs = json.dumps(
            [{sk: item[sk] for sk in schema_keys} for item in sample_output[:3]],
            indent=2,
        )  # Limit to 3 samples for brevity

        base_prompt = f"""Original prompt (that operates on the full input, not the individual chunks):
        {op_config['prompt']}
        
        Output schema:
        {json.dumps(op_config['output']['schema'], indent=2)}

        Sample inputs from processing various chunks:
        {sample_inputs}

        Modify the original prompt to be a prompt that will combine these chunk results to accomplish the original task. 

        Guidelines for your prompt template:
        - The only variable you are allowed to use is the values variable, which contains all chunk results. Each value is a dictionary with the keys {', '.join(schema_keys)}
        - Avoid using filters or complex logic, even though Jinja technically supports it
        - The prompt template must be a valid Jinja2 template
        - You must use the {{ values }} variable somehow (you can access specific schema keys if you'ld like)

        Provide your prompt template as a single string.
        """
        parameters = {
            "type": "object",
            "properties": {"combine_prompt": {"type": "string"}},
            "required": ["combine_prompt"],
        }

        result = generate_and_validate_prompt(
            self.llm_client,
            base_prompt,
            system_prompt,
            parameters,
            op_config,
            is_metadata=False,
            config=self.config,
            max_threads=self.max_threads,
            console=self.console,
        )
        combine_prompt = result["combine_prompt"]

        # Determine if the combine operation is commutative
        system_prompt_commutative = (
            "You are an AI assistant analyzing data processing tasks."
        )
        commutative_prompt = f"""
        Given the original task prompt and the combine prompt, determine if the order of combining chunk results matters.

        Original task prompt:
        {op_config['prompt']}
        
        Output schema:
        {json.dumps(op_config['output']['schema'], indent=2)}

        Sample inputs from processing various chunks:
        {sample_inputs}

        Prompt to combine results of subtasks:
        {combine_prompt}

        Does the order of combining chunk results matter? Answer with 'yes' if order matters (non-commutative) or 'no' if order doesn't matter (commutative). 
        Explain your reasoning briefly.

        For example:
        - Merging extracted key-value pairs from documents is commutative: combining {{"name": "John", "age": 30}} with {{"city": "New York", "job": "Engineer"}} yields the same result regardless of order
        - Generating a timeline of events is non-commutative: the order of events matters for maintaining chronological accuracy.

        Consider these examples when determining if the combining operation is commutative or not.
        """

        parameters_commutative = {
            "type": "object",
            "properties": {
                "is_commutative": {"type": "string", "enum": ["yes", "no"]},
                "explanation": {"type": "string"},
            },
            "required": ["is_commutative", "explanation"],
        }

        commutative_result = generate_and_validate_prompt(
            self.llm_client,
            commutative_prompt,
            system_prompt_commutative,
            parameters_commutative,
            op_config,
            is_metadata=False,
            config=self.config,
            max_threads=self.max_threads,
            console=self.console,
        )

        is_commutative = commutative_result["is_commutative"] == "no"
        commutative_explanation = commutative_result["explanation"]

        self.console.log("[bold]Commutativity Analysis:[/bold]")
        self.console.log(f"Is commutative: {'Yes' if is_commutative else 'No'}")
        self.console.log(f"Explanation: {commutative_explanation}")

        return combine_prompt, is_commutative

    def _edit_subprompt_to_reflect_metadata(
        self,
        subprompt: str,
        metadata_schema: Dict[str, Any],
        sample_output: List[Dict[str, Any]],
    ) -> str:
        # Select only metadata_schema keys from sample_output
        filtered_sample_output = []
        for item in sample_output:
            filtered_item = {key: item[key] for key in metadata_schema if key in item}
            filtered_sample_output.append(filtered_item)

        system_prompt = "You are an AI data processing agent. We have some metadata we can add to every document, and your job is to modify the data processing task prompt to reflect the new metadata."

        prompt = f"""
        Original task prompt:
        {subprompt}

        Metadata schema:
        {json.dumps(metadata_schema, indent=2)}

        Sample metadata output (from some docs):
        {json.dumps(filtered_sample_output[:3], indent=2)}

        Edit the original subprompt to incorporate the metadata. The new subprompt should:
        1. Reference the metadata fields where relevant
        2. Provide guidance on how to use the metadata in the context of the original task
        3. Maintain the original intent and requirements of the subprompt

        Provide the edited subprompt as a single string.
        """

        parameters = {
            "type": "object",
            "properties": {"edited_subprompt": {"type": "string"}},
            "required": ["edited_subprompt"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        return result["edited_subprompt"]
