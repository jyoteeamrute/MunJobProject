from py2neo import Node, Relationship, NodeMatcher
from constants import DEFAULT_PERSON_LABEL, DEFAULT_SKILL_LABEL, DEFAULT_NEW_SKILL_LABEL, MENU_ITEMS


class PersonManager:
    def __init__(self, graph, skill_manager, language="English"):
        self.graph = graph
        self.skill_manager = skill_manager
        self.matcher = NodeMatcher(graph)
        self.language = language

    def add_person(self, name, skills=None, warnings_fn=None):
        labels = MENU_ITEMS[self.language]

        existing_person = self.matcher.match(DEFAULT_PERSON_LABEL, name=name).first()
        if existing_person:
            message = labels["person_exists"].format(name=name)
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        person_node = Node(DEFAULT_PERSON_LABEL, name=name)
        self.graph.create(person_node)

        if skills:
            for skill in skills:
                skill_node = self.matcher.match(DEFAULT_SKILL_LABEL, title=skill['title']).first()
                if not skill_node:
                    skill_node = self.matcher.match(DEFAULT_NEW_SKILL_LABEL, title='title').first()
                if skill_node:
                    self.graph.create(Relationship(person_node, "HAS_SKILL", skill_node))
                else:
                    message = labels["skill_not_found"].format(skill_title=skill['title'])
                    print(message)
                    if warnings_fn:
                        warnings_fn(message)

        message = labels["person_added_success"].format(name=name)
        print(message)
        return True

    def connect_person_to_skills(self, person_name, skill_titles, warnings_fn=None):
        labels = MENU_ITEMS[self.language]

        person_node = self.matcher.match(DEFAULT_PERSON_LABEL, name=person_name).first()
        if not person_node:
            message = labels["person_not_exist"].format(person_name=person_name)
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        for skill_title in skill_titles:
            skill_node = self.matcher.match(DEFAULT_SKILL_LABEL, title=skill_title).first()
            if not skill_node:
                skill_node = self.matcher.match(DEFAULT_NEW_SKILL_LABEL, title=skill_title).first()
            if skill_node:
                self.graph.create(Relationship(person_node, skill_node))
            else:
                message = labels["skill_not_found"].format(skill_title=skill_title)
                print(message)
                if warnings_fn:
                    warnings_fn(message)

        message = labels["person_connected_to_skills"].format(person_name=person_name)
        print(message)
        return True

    def delete_person(self, name, warnings_fn=None):
        labels = MENU_ITEMS[self.language]

        person_node = self.matcher.match(DEFAULT_PERSON_LABEL, name=name).first()
        if not person_node:
            message = labels["person_not_exist_in_graph"].format(name=name)
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        self.graph.run("MATCH (p)-[r]-() WHERE id(p)=$id DELETE r", id=person_node.identity)
        self.graph.delete(person_node)

        message = labels["person_deleted"].format(name=name)
        print(message)
        return True

    def update_person(self, old_name, new_name, warnings_fn=None):
        labels = MENU_ITEMS[self.language]

        person_node = self.matcher.match(DEFAULT_PERSON_LABEL, name=old_name).first()
        if not person_node:
            message = labels["person_not_exist_in_graph"].format(name=old_name)
            print(message)
            warnings_fn(message)
            return False

        person_node["name"] = new_name
        self.graph.push(person_node)

        message = labels["person_updated"].format(name=new_name)
        print(message)
        return True

    def get_all_persons(self):
        persons = self.graph.run(f"MATCH (p:{DEFAULT_PERSON_LABEL}) RETURN p").data()
        return [person['p'].get('name', 'N/A') for person in persons]
