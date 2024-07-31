import pandas as pd
from py2neo import Node, Relationship, NodeMatcher
from constants import *


# Excel file fields / parameter
# Sl. / source_SI
# Id / source_Id
# Related_EN / title_en
# Skill_title_FI / title_fi
# Description / description_en

class ProfessionManager:
    def __init__(self, graph, gpt_client, embedding_manager, skill_manager):
        self.graph = graph
        self.gpt_client = gpt_client
        self.embedding_manager = embedding_manager
        self.skill_manager = skill_manager
        self.matcher = NodeMatcher(graph)

    def add_profession(self, title, title_fi, description=" ", skills=" ", description_fi=" ", source_sl=" ",
                       source_id=" ", warnings_fn=None):

        existing_profession = self.matcher.match(DEFAULT_PROFESSION_LABEL,
                                                 source_Id=source_id).first()

        if existing_profession:
            message = f"Profession {title} already exists in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return

        # maybe to use in production
        # skills = self.gpt_client.extract_skills_from_profession_description(title_description_en)
        skills = description

        profession_node = Node(DEFAULT_PROFESSION_LABEL,
                               source_Sl=source_sl,
                               source_Id=source_id,
                               title=title,
                               title_fi=title_fi,
                               description=description,
                               description_fi=description_fi,
                               skills=skills
                               )
        self.graph.create(profession_node)

        # self.embedding_manager.add_to_embeddings(title, skills, 'Professions', warnings_fn=warnings_fn)

        filtered_skills = self.embedding_manager.filter_skills(title=title, description=skills,
                                                               warnings_fn=warnings_fn)

        matched_skills = self.gpt_client.match_skills_for_profession(filtered_skills, f"{title}::{skills}")

        for pair in matched_skills:
            extracted_skill = pair['extracted_skill']
            matched_skill = pair['common_skill']

            if matched_skill.lower() == 'new':
                matched_skill, relationship_type = self.skill_manager.handle_new_skill(extracted_skill)
            else:
                relationship_type = f"EXACT [{extracted_skill}]"

            profession_node = self.matcher.match(DEFAULT_PROFESSION_LABEL, title=title).first()

            if relationship_type == 'NEW':
                skill_node = self.matcher.match(DEFAULT_NEW_SKILL_LABEL, title=matched_skill).first()
            else:
                skill_node = self.matcher.match(DEFAULT_SKILL_LABEL, title=matched_skill).first()

            if profession_node and skill_node:
                rel = Relationship(profession_node, "REQUIRES_SKILL", skill_node, type=relationship_type)
                self.graph.create(rel)
            else:
                message = f"Node {matched_skill} or node {skill_node} not found"
                print(message)
                if warnings_fn:
                    warnings_fn(message)

    def add_professions_batch(self, professions_batch, warnings_fn=None):
        for profession_data in professions_batch:
            self.add_profession(
                source_sl=profession_data['source_sl'],
                source_id=profession_data['source_id'],
                title=profession_data['title'],
                title_fi=profession_data['title_fi'],
                description=profession_data['description'],
                warnings_fn=warnings_fn
            )

    def load_professions_from_file(self, file_path, batch_size=BATCH_SIZE, warnings_fn=None, progress_fn=None):

        try:
            df = pd.read_excel(file_path)
            required_columns = ['Sl.', 'Id', 'Related_EN', 'Skill_title_EN', 'Skill_title_FI', 'Description']
            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            # Excel file fields / parameter
            # Sl. / source_Sl
            # Id / source_Id
            # Related_EN / title
            # Skill_title_FI / title_fi
            # Description / description

            # Initialize the progress bar
            total_rows = len(df)
            if progress_fn:
                progress_step = 100 / total_rows if total_rows else 1

            # Prepare batch processing
            professions_batch = []

            # Iterate over rows and add each profession to the batch
            for index, row in df.iterrows():
                profession_data = {
                    'source_sl': row['Sl.'],
                    'source_id': row['Id'],
                    'title': row['Related_EN'],
                    'title_fi': row['Skill_title_FI'],
                    'description': row['Description']
                }

                professions_batch.append(profession_data)

                # Update progress bar
                if progress_fn:
                    progress_fn(min((index + 1) * progress_step, 100))

                # When batch size is reached, process the batch
                if len(professions_batch) >= batch_size:
                    self.add_professions_batch(professions_batch, warnings_fn)
                    professions_batch.clear()  # Clear the batch

            # Process any remaining professions in the last batch
            if professions_batch:
                self.add_professions_batch(professions_batch, warnings_fn)


        except Exception as e:
            message = f"Error loading professions from file: {e}"
            print(message)
            if warnings_fn:
                warnings_fn(message)

    def delete_profession(self, profession_title, warnings_fn=None):
        profession_node = self.matcher.match(DEFAULT_PROFESSION_LABEL, title=profession_title).first()

        if not profession_node:
            message = f"Profession '{profession_title}' does not exist in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return

        title = profession_node.get('title', 'none')
        skills = profession_node.get('skills', 'none')

        # Delete all relationships associated with the profession
        self.graph.run("MATCH (p)-[r]-() WHERE id(p)=$id DELETE r", id=profession_node.identity)

        # Remove the profession node
        self.graph.delete(profession_node)

        # Delete associated embeddings
        # self.embedding_manager.delete_from_embeddings(title, skills, 'Professions', warnings_fn)

    def update_profession(self, profession_title, **kwargs):
        profession_node = self.matcher.match(DEFAULT_PROFESSION_LABEL, title=profession_title).first()

        if not profession_node:
            message = f"Profession '{profession_title}' does not exist in the graph."
            print(message)

            return

        # old_profession_title = profession_node['title']
        # old_profession_skills = profession_node['skills']

        updated_fields = []
        for key, value in kwargs.items():
            if key != 'warnings_fn' and value is not None:
                profession_node[key] = value
                updated_fields.append(key)

        self.graph.push(profession_node)

        if 'title' in updated_fields or 'skills' in updated_fields:
            new_profession_title = profession_node['title']
            new_profession_skills = profession_node['skills']
            # self.embedding_manager.update_embeddings(old_profession_title, old_profession_skills, new_profession_title, new_profession_skills,'Professions', warning_fn=warnings_fn)

    def get_all_professions(self):
        professions = self.graph.run(f"MATCH (p:{DEFAULT_PROFESSION_LABEL}) RETURN p").data()

        profession_list = [
            {
                'source_sl': profession['p'].get('source_sl', ''),
                'source_id': profession['p'].get('source_id', ''),
                'title': profession['p'].get('title', ''),
                'title_fi': profession['p'].get('title_fi', ''),
                'description': profession['p'].get('description', ''),
                'skills': profession['p'].get('skills', '')
            }
            for profession in professions
        ]

        return profession_list

    def connect_profession_to_skills(self, profession_title, selected_skills, warnings_fn=None):
        profession_node = self.matcher.match(DEFAULT_PROFESSION_LABEL, title=profession_title).first()

        if not profession_node:
            message = f"Profession '{profession_title}' does not exist in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        for skill_title in selected_skills:

            skill_node = self.matcher.match(DEFAULT_SKILL_LABEL, title=skill_title).first()
            if not skill_node:
                skill_node = self.matcher.match(DEFAULT_NEW_SKILL_LABEL, title=skill_title).first()

            if skill_node:
                relationship_type = "REQUIRES_SKILL"
                rel = Relationship(profession_node, "REQUIRES_SKILL", skill_node, type=relationship_type)
                self.graph.create(rel)
            else:
                message = f"Skill '{skill_title}' does not exist in the graph."
                print(message)
                if warnings_fn:
                    warnings_fn(message)

        return True
