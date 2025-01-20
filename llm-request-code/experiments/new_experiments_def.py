from req_smell_tracing.experiment import Experiment, Generators
from req_smell_tracing.llm import GPT, Ollama, Prompt


def def_experiments(experiments, games, config):
    if 'local' in config['LLM_NAME']:
        llm = Ollama(
            config,
            model="llama3.1",
            temperature=0.3,
            response_format="json"
        )
    else:
        llm = GPT(
            config,
            model="gpt-4o",
            temperature=0,
            response_format="json",
        )

    prompts = [
        Prompt(
            config,
            role="system",
            content_file="system_prompt_guidelines.txt",
        ),
        Prompt(
            config,
            role="user",
            content_file="user_prompt.txt",
        ),
    ]

    base = Experiment(
        description="Tracing with a random number of smells.",
        code_language="java",
        iterations=3,
        llm=llm,
        prompts=prompts
    )

    # ===== No Smells =====

    bases = []
    bases.append(
        base.alter_new(name="no_smells", group="dice_game", game=games["dice_game"])
    )
    bases.append(
        base.alter_new(name="no_smells", group="arkanoid", game=games["arkanoid"])
    )
    bases.append(base.alter_new(name="no_smells", group="snake", game=games["snake"]))
    bases.append(base.alter_new(name="no_smells", group="scopa", game=games["scopa"]))
    bases.append(base.alter_new(name="no_smells", group="pong", game=games["pong"]))

    for base in bases:
        experiments.extend(base.from_generator(Generators.no_smells, variations=1))

    # # ===== All Smells ===== 
    bases = []
    bases.append(
        base.alter_new(name="all_smells", group="dice_game", game=games["dice_game"])
    )
    bases.append(
        base.alter_new(name="all_smells", group="arkanoid", game=games["arkanoid"])
    )
    bases.append(base.alter_new(name="all_smells", group="snake", game=games["snake"]))
    bases.append(base.alter_new(name="all_smells", group="scopa", game=games["scopa"]))
    bases.append(base.alter_new(name="all_smells", group="pong", game=games["pong"]))
    for base in bases:
        experiments.extend(base.from_generator(Generators.all_smells, variations=1))
    
    # ===== Random Smells =====
    bases = []
    bases.append(base.alter_new(group="dice_game", game=games["dice_game"]))
    bases.append(base.alter_new(group="arkanoid", game=games["arkanoid"]))
    bases.append(base.alter_new(group="snake", game=games["snake"]))
    bases.append(base.alter_new(group="scopa", game=games["scopa"]))
    bases.append(base.alter_new(group="pong", game=games["pong"]))

    for base in bases:
        experiments.extend(
            base.alter_new(name="random").from_generator(
                Generators.random_smells, variations=2
            )
        )

    # ===== Smell Type Specific Smells =====
    bases = []
    bases.append(base.alter_new(group="dice_game", game=games["dice_game"]))
    bases.append(base.alter_new(group="arkanoid", game=games["arkanoid"]))
    bases.append(base.alter_new(group="snake", game=games["snake"]))
    bases.append(base.alter_new(group="scopa", game=games["scopa"]))
    bases.append(base.alter_new(group="pong", game=games["pong"]))

    for base in bases:
        experiments.extend(
            base.alter_new(name="only_lexical_smells").from_generator(
                Generators.smell_type_specific_smells, variations=1, smell_type=1, game=base.game
            )
        )
        experiments.extend(
            base.alter_new(name="only_semantic_smells").from_generator(
                Generators.smell_type_specific_smells, variations=1, smell_type=2, game=base.game
            )
        )
        experiments.extend(
            base.alter_new(name="only_syntactic_smells").from_generator(
                Generators.smell_type_specific_smells, variations=1, smell_type=3, game=base.game
            )
        )

    debugger_test = "test"
