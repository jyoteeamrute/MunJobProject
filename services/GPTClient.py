import openai
import json
import os
from openai import OpenAIError


class GPTClient:
    def __init__(self, model="gpt-4o"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def chat(self, system_message, user_message, additional_messages=None, warnings_fn=None):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        if additional_messages:
            messages.extend(additional_messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            print("RESPONSE:", response.choices[0].message.content)

            return response.choices[0].message.content

        except OpenAIError as e:
            if warnings_fn:
                warnings_fn(f"Failed to generate response: {e}")
            return None

    @staticmethod
    def remove_quotes(text, warnings_fn=None):
        if isinstance(text, str):
            text = text.replace('"', '').replace("'", '')
            return " ".join(text.strip().split())
        return ""

    def generate_skill_description_english(self, skill_title, course_and_profession_info, warnings_fn=None):
        # Remove any quotation marks for cleaner input processing.
        skill_title = self.remove_quotes(skill_title)
        course_and_profession_info = self.remove_quotes(course_and_profession_info)
        # Define the AI's role clearly as an expert in generating detailed skill descriptions using the provided contextual information.
        system_message = "You are an expert tasked with creating detailed skill descriptions in English."
        # Direct the AI to provide a description that incorporates details about the courses that teach this skill and the professions that require it.
        user_message = f"Generate a detailed and concise description for the skill: '{skill_title}', using the additional information if provided about the courses and professions involved. Additional information {course_and_profession_info} Limit the response to three sentences."
        return self.chat(system_message, user_message)

    def generate_skill_description_finnish(self, skill_title, course_and_profession_info, warnings_fn=None):
        skill_title = self.remove_quotes(skill_title)
        course_and_profession_info = self.remove_quotes(course_and_profession_info)
        system_message = "As a Finnish-speaking expert, generate a detailed yet concise description for the following skill."
        user_message = f"Skill: {skill_title}. Relevant courses and professions: {course_and_profession_info}. Please provide the description in Finnish, up to three sentences."
        return self.chat(system_message, user_message)

    def translate(self, text, target_language="English", warnings_fn=None):
        # Removes any non-essential conversational elements or personalizations.
        text = self.remove_quotes(text)
        # Direct, professional instruction aimed specifically at translation.
        system_message = "You are a professional translator. Provide a direct translation."
        # Request a literal translation of the text to the specified language.
        user_message = f"Translate the following text to {target_language}: '{text}'. Provide only translation.No comments from your side."
        return self.chat(system_message, user_message)

    def extract_skills_from_course_description(self, description, warnings_fn=None):
        # Remove any quotation marks from the course description for clearer input.
        description = self.remove_quotes(description)
        # Clearly define the task for the AI, focusing on identifying skills gained from the course.
        system_message = "You are a learning and development expert. Your task is to identify skills that are developed by taking this course."
        # Ask the AI to list the skills acquired from the course and provide a concise description for each skill.
        user_message = f"From the following course description, identify and describe each skill that will be acquired by completing the course: '{description}' Please provide a brief description for each identified skill."
        return self.chat(system_message, user_message)

    def extract_skills_from_profession_description(self, description, warnings_fn=None):
        description = self.remove_quotes(description)
        system_message = "You are a career development expert. Your task is to identify skills that are essential for the given profession."
        user_message = f"From the following profession description, identify and describe each skill that is essential for performing well in this profession: '{description}' Please provide a brief description for each identified skill. Only skills with very short descriptions. No comments from your side."
        return self.chat(system_message, user_message)

    def match_skills_for_course(self, filtered_skills, combined_description, warnings_fn=None):
        combined_description = self.remove_quotes(combined_description)
        # filtered_skills = self.remove_quotes(filtered_skills)
        system_message = "You are a professional experienced learning and development expert."
        user_message = (
            "You are provided with two lists:\n"
            "1. A common list of skills with titles and descriptions.\n"
            "2. A combined title and description of skills that can be gained by completing a specific training program.\n"
            "For each skill mentioned in the program's combined description:\n"
            "1. Identify the exact matching skill from the common skills list.\n"
            "2. If there is an very close match, pair the extracted skill with the matching skill from the common list.\n"
            "3. If there is no very close match, pair the extracted skill with 'new'.\n"
            "The response should be formatted as a JSON object:\n"
            "[{\"extracted_skill\": \"skill_title\", \"common_skill\": \"matched_skill_or_new\"}, ...]\n"
            f"Common skills list: {json.dumps(filtered_skills)}.\nProgram Combined Description: {combined_description}"

        )

        response = self.chat(system_message, user_message)
        clean_answer = response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean_answer)
        except json.JSONDecodeError:
            if warnings_fn:
                warnings_fn("Failed to decode JSON response from gpt. Courses skill matching error.")
            return []

    def match_skills_for_profession(self, filtered_skills, profession_description, warnings_fn=None):
        # Clean the profession description by removing quotes
        profession_description = self.remove_quotes(profession_description)

        # Define the system message for the GPT model
        system_message = (
            "You are a professional experienced in career development and skill assessment. "
            "Your expertise includes identifying and matching relevant skills required for various professions."
        )

        # Define the user message with instructions for the GPT model
        user_message = (
            "You are provided with two documents:\n"
            "1. THE MAIN  SKILL LIST - list of skills with titles and description.\n"
            "2. THE DETAILED PROFESSION'S DESCRIPTION of a specific profession outlining the skills required for that profession.\n"
            "For each skill mentioned in THE DETAILED PROFESSION'S DESCRIPTION:\n"
            "1. Identify the EXACT matching skill from THE MAIN SKILL LIST.\n"
            "2. If there is exact match, pair the skill extracted from THE DETAILED PROFESSION'S DESCRIPTION with the exact matching skill from THE MAIN SKILL LIST.\n"
            "3. If there is no exact match, pair the extracted skill from THE DETAILED PROFESSION'S DESCRIPTION with 'new'.\n"
            "The response should be formatted as a JSON object:\n"
            "[{\"extracted_skill\": \"skill_title\", \"common_skill\": \"matched_skill_or_new\"}, ...]\n"
            f"THE MAIN SKILL LIST: {filtered_skills}"
            f"THE PROFESSION'S DETAILED DESCRIPTION: {profession_description}")

        # Send the messages to the GPT model and get the response
        response = self.chat(system_message, user_message)

        # Clean the response by removing any unwanted formatting
        clean_answer = response.replace("```json", "").replace("```", "").strip()

        # Attempt to parse the response as JSON and handle any errors
        try:
            return json.loads(clean_answer)
        except json.JSONDecodeError:
            if warnings_fn:
                warnings_fn("Failed to decode JSON response from GPT. Profession skill matching error.")
            return []
