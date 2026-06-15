"""Generate a complete Hydra learning repository.

Run this script in Google Colab or locally to create a progressive tutorial that
covers Hydra basics, config groups, overrides, interpolation, structured configs,
object instantiation, multirun, plugins, and a realistic ML project.
"""

import os
import subprocess  # noqa: S404 - used only to install declared tutorial dependencies.
import sys
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path('hydra_tutorial')


def write(path: str, content: str) -> None:
    """Write dedented content to a generated tutorial file."""
    full = ROOT / path
    full.parent.mkdir(exist_ok=True, parents=True)
    with full.open('w', encoding='utf-8') as f:
        f.write(textwrap.dedent(content).lstrip('\n'))


def append_file(path: str, content: str) -> None:
    """Append dedented text to a generated file inside ROOT."""
    full = ROOT / path
    full.parent.mkdir(exist_ok=True, parents=True)
    with full.open('a', encoding='utf-8') as f:
        f.write('\n' + textwrap.dedent(content).lstrip('\n'))


def banner(text: str, char: str = '═') -> None:
    """Print a tutorial banner."""
    width = 80
    print('\n' + char * width)
    print(f'  {text}')
    print(char * width)


def section(title: str) -> None:
    """Print a tutorial section heading."""
    print(f'\n{"─" * 70}')
    print(f'  📁  {title}')
    print('─' * 70)


def note(text: str) -> None:
    """Print wrapped explanatory text."""
    for line in textwrap.wrap(text, 76):
        print(f'  {line}')


def cmd(text: str) -> None:
    """Print a shell command prompt."""
    print(f'\n  \033[92m$\033[0m  {text}')


def install_hydra() -> None:
    """Install Hydra and the tutorial launcher dependency."""
    banner('STEP 0 — Installing hydra-core & omegaconf')
    # --break-system-packages needed on Colab / Ubuntu 24+ / PEP 668 envs
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            'hydra-core',
            'omegaconf',
            'hydra-joblib-launcher',
            '--quiet',
            '--break-system-packages',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Fallback without the flag (older pip / virtualenv)
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'hydra-core', 'omegaconf', 'hydra-joblib-launcher', '--quiet'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print('  ✅  hydra-core installed successfully')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 0 — BASICS
# ─────────────────────────────────────────────────────────────────────────────


def create_00_basics() -> None:
    """Create the introductory Hydra lesson."""
    section('00_basics — Your first Hydra app')

    write(
        '00_basics/conf/config.yaml',
        """
        # The root config file.
        # Hydra reads this when you run the app.
        # Access values with dot notation: cfg.db.host

        db:
          host: localhost
          port: 5432
          name: my_database
          user: admin

        training:
          lr: 0.001
          epochs: 10
          batch_size: 32
          device: cpu
    """,
    )

    write(
        '00_basics/app.py',
        """
        \"\"\"
        LESSON 0 — The simplest possible Hydra app.

        Key concepts:
          • @hydra.main decorator wires config loading to your function
          • config_path points to the folder containing YAML files
          • config_name is the YAML filename WITHOUT .yaml
          • version_base=None silences future-version warnings
          • cfg is an OmegaConf DictConfig — access with dot notation
        \"\"\"
        import hydra
        from omegaconf import DictConfig, OmegaConf

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig) -> None:
            # ── Print the entire resolved config ──────────────────────────────
            print("\\n=== Full config (as YAML) ===")
            print(OmegaConf.to_yaml(cfg))

            # ── Access individual values ───────────────────────────────────────
            print(f"DB host  : {cfg.db.host}")
            print(f"DB port  : {cfg.db.port}")          # int, not string
            print(f"LR       : {cfg.training.lr}")      # float
            print(f"Epochs   : {cfg.training.epochs}")

            # ── Convert to plain Python dict when needed ───────────────────────
            plain_dict = OmegaConf.to_container(cfg, resolve=True)
            print(f"\\nType of cfg         : {type(cfg)}")
            print(f"Type of plain_dict  : {type(plain_dict)}")

        if __name__ == "__main__":
            main()
    """,
    )

    write(
        '00_basics/LESSON.md',
        """
        # Lesson 0 — Basics

        ## Run the app (default config)
        ```
        cd 00_basics
        python app.py
        ```

        ## Override any value from the CLI
        ```
        python app.py db.host=prod-server db.port=5433
        python app.py training.lr=0.01 training.epochs=50
        python app.py training.device=cuda
        ```

        ## Print config without running the app
        ```
        python app.py --cfg job
        ```

        ## Key facts
        - `cfg` is read-only by default (OmegaConf struct mode).
        - Dot notation: `cfg.db.host`  — same as `cfg["db"]["host"]`
        - Types are preserved from YAML: port=5432 is int, lr=0.001 is float.
        - Hydra auto-creates `outputs/YYYY-MM-DD/HH-MM-SS/` and chdirs there.
          Your script's CWD is that outputs dir when running.
    """,
    )

    note('Created: 00_basics/ with app.py + conf/config.yaml')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CONFIG GROUPS
# ─────────────────────────────────────────────────────────────────────────────


def create_01_config_groups() -> None:
    """Create the configuration-groups lesson."""
    section('01_config_groups — Swapping config blocks')

    # ── db group ──
    write(
        '01_config_groups/conf/db/postgres.yaml',
        """
        # @package _global_
        db:
          driver: postgresql
          host: localhost
          port: 5432
          user: postgres
          password: secret
          pool_size: 10
    """,
    )
    write(
        '01_config_groups/conf/db/sqlite.yaml',
        """
        # @package _global_
        db:
          driver: sqlite
          path: /tmp/app.db
          pool_size: 1
    """,
    )
    write(
        '01_config_groups/conf/db/mysql.yaml',
        """
        # @package _global_
        db:
          driver: mysql
          host: localhost
          port: 3306
          user: root
          password: root
          pool_size: 5
    """,
    )

    # ── model group ──
    write(
        '01_config_groups/conf/model/small.yaml',
        """
        # @package _global_
        model:
          name: transformer_small
          hidden_size: 128
          num_layers: 2
          num_heads: 4
          dropout: 0.1
          max_seq_len: 512
    """,
    )
    write(
        '01_config_groups/conf/model/medium.yaml',
        """
        # @package _global_
        model:
          name: transformer_medium
          hidden_size: 512
          num_layers: 8
          num_heads: 8
          dropout: 0.2
          max_seq_len: 1024
    """,
    )
    write(
        '01_config_groups/conf/model/large.yaml',
        """
        # @package _global_
        model:
          name: transformer_large
          hidden_size: 1024
          num_layers: 24
          num_heads: 16
          dropout: 0.1
          max_seq_len: 2048
    """,
    )

    # ── optimizer group ──
    write(
        '01_config_groups/conf/optimizer/adam.yaml',
        """
        # @package _global_
        optimizer:
          name: adam
          lr: 0.001
          betas: [0.9, 0.999]
          eps: 1.0e-8
          weight_decay: 0.0
    """,
    )
    write(
        '01_config_groups/conf/optimizer/sgd.yaml',
        """
        # @package _global_
        optimizer:
          name: sgd
          lr: 0.01
          momentum: 0.9
          weight_decay: 1.0e-4
          nesterov: true
    """,
    )
    write(
        '01_config_groups/conf/optimizer/adamw.yaml',
        """
        # @package _global_
        optimizer:
          name: adamw
          lr: 3.0e-4
          betas: [0.9, 0.95]
          eps: 1.0e-8
          weight_decay: 0.1
    """,
    )

    # ── root config ──
    write(
        '01_config_groups/conf/config.yaml',
        """
        defaults:
          - db: postgres          # loads conf/db/postgres.yaml
          - model: small          # loads conf/model/small.yaml
          - optimizer: adam       # loads conf/optimizer/adam.yaml
          - _self_                # this file's keys merge last

        experiment_name: baseline
        seed: 42
        debug: false
    """,
    )

    write(
        '01_config_groups/train.py',
        """
        \"\"\"
        LESSON 1 — Config groups: swap entire config blocks at runtime.

        The defaults list in config.yaml says:
          - db: postgres   →  loads conf/db/postgres.yaml under key 'db'
          - model: small   →  loads conf/model/small.yaml under key 'model'

        You can override the CHOICE of group from the CLI:
          python train.py db=sqlite         (use sqlite.yaml instead)
          python train.py model=large       (use large.yaml instead)
          python train.py optimizer=adamw   (use adamw.yaml instead)
        \"\"\"
        import hydra
        from omegaconf import DictConfig, OmegaConf

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def train(cfg: DictConfig) -> None:
            print("\\n" + "="*60)
            print(f"Experiment : {cfg.experiment_name}")
            print(f"Seed       : {cfg.seed}")
            print()
            print(f"Database   : {cfg.db.driver} @ {cfg.db.get('host', cfg.db.get('path'))}")
            print(f"Model      : {cfg.model.name}  ({cfg.model.hidden_size}d x {cfg.model.num_layers}L)")
            print(f"Optimizer  : {cfg.optimizer.name}  (lr={cfg.optimizer.lr})")
            print()
            print("--- Full merged config ---")
            print(OmegaConf.to_yaml(cfg))

        if __name__ == "__main__":
            train()
    """,
    )

    write(
        '01_config_groups/LESSON.md',
        """
        # Lesson 1 — Config Groups

        ## Default (postgres + small + adam)
        ```
        cd 01_config_groups
        python train.py
        ```

        ## Swap the database
        ```
        python train.py db=sqlite
        python train.py db=mysql
        ```

        ## Swap model size
        ```
        python train.py model=large
        python train.py model=medium
        ```

        ## Mix and match freely
        ```
        python train.py db=sqlite model=large optimizer=adamw
        python train.py db=mysql model=medium optimizer=sgd experiment_name=my_exp
        ```

        ## Override a value *within* the chosen group
        ```
        python train.py model=large model.dropout=0.05
        python train.py optimizer=adam optimizer.lr=3e-4
        ```

        ## Print which config files were loaded
        ```
        python train.py --info defaults
        ```

        ## Key concepts
        - Config group identifier = folder name, e.g. `db`, `model`, `optimizer`.
        - Config option identifier = file name without `.yaml`, e.g. `postgres`.
        - Identifiers usually are not inside the YAML file; Hydra discovers them from paths.
        - `# @package _global_` means merge the file at the root of the final config tree.
        - Without `_global_`, Hydra normally places a group config under its group path.
        - `_self_` controls merge precedence, not hierarchy. Later values win on overlap.
        - Avoid overlapping ownership: let `optimizer/` own `optimizer.*`, `model/` own `model.*`, etc.
    """,
    )

    note('Created: 01_config_groups/ with db/model/optimizer groups')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — OVERRIDES (deep dive)
# ─────────────────────────────────────────────────────────────────────────────


def create_02_overrides() -> None:
    """Create the command-line overrides lesson."""
    section('02_overrides — Every override syntax explained')

    write(
        '02_overrides/conf/config.yaml',
        """
        app:
          name: myapp
          debug: false
          tags: [production, v1]
          nested:
            level1:
              level2: deep_value

        limits:
          max_retries: 3
          timeout: 30.0

        server:
          host: 0.0.0.0
          port: 8080
    """,
    )

    write(
        '02_overrides/demo.py',
        """
        \"\"\"
        LESSON 2 — CLI override operators and shell quoting.

        Override operators:
          key=value       update an existing key
          +key=value      add a NEW key only; errors if the key already exists
          ++key=value     upsert: update if it exists, add if it does not
          ~key            delete a key

        Important:
          Hydra parses the override AFTER your shell is done interpreting it.
          Quoting protects special characters from bash/zsh; it usually does
          not change Hydra's meaning.
        \"\"\"
        import hydra
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import DictConfig, OmegaConf, open_dict

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def demo(cfg: DictConfig) -> None:
            print("\\n=== Final job config ===")
            print(OmegaConf.to_yaml(cfg))
            print("--- Overrides Hydra received ---")
            print(list(HydraConfig.get().overrides.task))

            with open_dict(cfg):
                cfg.app.runtime_id = "abc123"
            print(f"\\nruntime_id added in Python: {cfg.app.runtime_id}")

        if __name__ == "__main__":
            demo()
    """,
    )

    write(
        '02_overrides/LESSON.md',
        r"""
        # Lesson 2 — Override Syntax and Shell Quoting

        Run all commands from `02_overrides/`.

        ## Shell quoting equivalences

        These pairs produce the same Hydra config in bash/zsh:

        ```bash
        python demo.py 'app.tags=[staging,v2,hotfix]'
        python demo.py app.tags="[staging,v2,hotfix]"
        ```

        ```bash
        python demo.py 'app.name="hello:world"'
        python demo.py app.name="hello:world"
        python demo.py 'app.name=hello:world'
        ```

        ```bash
        python demo.py '+server={host:127.0.0.1,port:9090,ssl:true}'
        python demo.py +server="{host:127.0.0.1,port:9090,ssl:true}"
        ```

        The shell removes its own quote characters before Python receives the
        arguments. The quotes mostly protect brackets/braces/commas/colons from
        the shell.

        ## `+` vs `++`

        | Operator | If key exists | If key does not exist | Meaning |
        |----------|---------------|-----------------------|---------|
        | `key=value` | Updates it | Error | Modify existing key only |
        | `+key=value` | Error | Adds it | Insert new key only |
        | `++key=value` | Updates it | Adds it | Upsert: update or insert |

        This table is easiest to understand for leaf paths like `app.name`.
        Whole-dictionary overrides such as `+server={...}` can merge/update
        dictionary contents while adding new subkeys.

        Try:

        ```bash
        python demo.py app.name=new_name       # works: existing key
        python demo.py +app.version=2.0       # works: new key
        python demo.py ++app.version=2.0      # works: new key
        python demo.py ++app.name=new_name    # works: existing key
        ```

        These should fail:

        ```bash
        python demo.py +app.name=new_name     # app.name already exists
        python demo.py app.version=2.0        # app.version does not exist
        ```

        ## Inline dictionary syntax

        Inside Hydra dictionaries, use `:` between key and value:

        ```bash
        python demo.py '+server={host:127.0.0.1,port:9090,ssl:true}'
        python demo.py +server="{host:127.0.0.1,port:9090,ssl:true}"
        ```

        This works as a dictionary merge/update on `server`: existing fields
        like `host` and `port` change, and the new `ssl` field is added.

        This is wrong inside the dictionary:

        ```bash
        python demo.py '+server={host:127.0.0.1,port:9090,ssl=true}'
        ```

        For leaf keys, `+` is stricter:

        ```bash
        python demo.py +app.name=new_name       # fails because app.name exists
        python demo.py +server.host=127.0.0.1   # fails because server.host exists
        ```

        ## More examples

        ```bash
        python demo.py '~limits.max_retries'
        python demo.py app.debug=null
        python demo.py 'app.tags=[staging,v2,hotfix]'
        python demo.py --cfg job
        python demo.py --cfg hydra
        python demo.py --cfg all
        ```

        `--cfg job` means: print the application/job config and exit.
    """,
    )

    note('Created: 02_overrides/ — override operators, quoting, + vs ++')


def create_03_interpolation() -> None:
    """Create the configuration interpolation lesson."""
    section('03_interpolation — Variable references & resolvers')

    write(
        '03_interpolation/conf/config.yaml',
        """
        project: hydra_demo
        version: "1.0"
        env: development

        # Stored formula-style; resolved when accessed.
        run_name: "${project}_${version}_${env}"

        base_dir: /workspace
        data_dir: ${base_dir}/data
        output_dir: ${base_dir}/outputs/${project}
        checkpoint_dir: ${output_dir}/checkpoints

        db:
          host: db-server
          port: 5432
          name: mydb
          url: "postgresql://${db.host}:${db.port}/${db.name}"

        model:
          name: resnet50
          save_path: "${checkpoint_dir}/${model.name}.pt"

        # oc.env is a built-in OmegaConf resolver for OS environment variables.
        api_key: ${oc.env:API_KEY,not_set}
        home_dir: ${oc.env:HOME,/root}

        # oc.select safely reads a missing key with fallback.
        safe_missing_example: ${oc.select:maybe.missing.key,fallback_value}

        hyperparams:
          lr: 0.001
          epochs: 10
          batch: 32

        # Calls a Python resolver registered in app.py. It does not select a file.
        log_level: ${choose_log_level:${env}}
    """,
    )

    write(
        '03_interpolation/app.py',
        """
        \"\"\"
        LESSON 3 — Interpolation and OmegaConf resolvers.

        Lazy means formula-like:
          ${base_dir}/data is not permanently computed at load time.
          It resolves when cfg.data_dir is accessed.

        oc.env is a built-in resolver for OS environment variables.
        Custom resolvers are Python functions registered before @hydra.main.
        \"\"\"
        import hydra
        from omegaconf import DictConfig, OmegaConf, open_dict

        OmegaConf.register_new_resolver(
            "choose_log_level",
            lambda env: "DEBUG" if env == "development" else "INFO"
        )
        OmegaConf.register_new_resolver("mul", lambda a, b: float(a) * float(b))
        OmegaConf.register_new_resolver("upper", lambda s: str(s).upper())

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig) -> None:
            print("\\n=== Interpolation Demo ===\\n")
            print(f"run_name           : {cfg.run_name}")
            print(f"data_dir           : {cfg.data_dir}")
            print(f"output_dir         : {cfg.output_dir}")
            print(f"checkpoint_dir     : {cfg.checkpoint_dir}")
            print(f"model.save_path    : {cfg.model.save_path}")
            print(f"db.url             : {cfg.db.url}")
            print(f"api_key            : {cfg.api_key}")
            print(f"home_dir           : {cfg.home_dir}")
            print(f"safe_missing       : {cfg.safe_missing_example}")
            print(f"log_level          : {cfg.log_level}")

            print("\\n--- Lazy resolution demo ---")
            print("Before changing cfg.project: output_dir =", cfg.output_dir)
            with open_dict(cfg):
                cfg.project = "NEW_PROJECT"
            print("After  changing cfg.project: output_dir =", cfg.output_dir)
            print("run_name also re-resolves   :", cfg.run_name)

            print("\\n--- Custom resolver demo ---")
            result = OmegaConf.create({
                "area": "${mul:${w},${h}}",
                "title": "${upper:${name}}",
                "w": 10,
                "h": 5,
                "name": "hydra",
            })
            print(f"10 x 5 = {result.area}")
            print(f"upper('hydra') = {result.title}")

        if __name__ == "__main__":
            main()
    """,
    )

    write(
        '03_interpolation/LESSON.md',
        r"""
        # Lesson 3 — Interpolation

        ## Basic run

        ```bash
        cd 03_interpolation
        python app.py
        ```

        ## Lazy means formula-like

        ```yaml
        base_dir: /workspace
        output_dir: ${base_dir}/outputs/${project}
        ```

        `output_dir` is not permanently computed when the config is loaded.
        It is resolved when you access `cfg.output_dir`.

        ```bash
        python app.py base_dir=/data/experiments
        ```

        `data_dir`, `output_dir`, `checkpoint_dir`, and `model.save_path`
        update because they reference `base_dir`.

        ## `oc.env` means OS environment variable

        ```yaml
        api_key: ${oc.env:API_KEY,not_set}
        ```

        `oc.env` is a built-in OmegaConf resolver. It reads the OS environment.
        `${project}` reads from the config tree.

        ```bash
        API_KEY=mysecretkey python app.py
        ```

        ## Custom resolver: `choose_log_level`

        ```yaml
        env: development
        log_level: ${choose_log_level:${env}}
        ```

        This calls the Python function registered in `app.py`:

        ```python
        lambda env: "DEBUG" if env == "development" else "INFO"
        ```

        It does not select another config file.

        ```bash
        python app.py env=production
        python app.py env=development
        ```

        ## Deleting an interpolation target

        ```bash
        python app.py '~project'
        ```

        `~project` deletes the key. Then `run_name` tries to resolve
        `${project}` and errors when accessed. `-m` is not needed here.

        ## `oc.select` for safe access

        ```yaml
        safe_value: ${oc.select:maybe.missing.key,fallback_value}
        ```

        This returns `fallback_value` instead of crashing if the key is missing.
    """,
    )

    note('Created: 03_interpolation/ with clearer resolver and lazy examples')


def create_04_structured_configs() -> None:
    """Create the structured-configs lesson."""
    section('04_structured_configs — Typed configs with dataclasses')

    write(
        '04_structured_configs/schemas.py',
        """
        \"\"\"
        Structured config schemas using Python dataclasses.

        Mental model:
          YAML values      = actual experiment values
          AppConfig schema = allowed tree shape + types + defaults
          MISSING          = required value; no sensible default

        Registering the schema is not enough. The YAML must include it in the
        defaults list to activate validation.
        \"\"\"
        from dataclasses import dataclass, field
        from typing import List, Optional
        from omegaconf import MISSING

        @dataclass
        class DBConfig:
            host: str = "localhost"
            port: int = 5432
            name: str = MISSING
            user: str = "admin"
            password: str = MISSING

        @dataclass
        class ModelConfig:
            name: str = "baseline"
            hidden_size: int = 256
            num_layers: int = 4
            dropout: float = 0.1
            activation: str = "relu"

        @dataclass
        class OptimizerConfig:
            lr: float = 1e-3
            weight_decay: float = 0.0
            clip_grad_norm: Optional[float] = None

        @dataclass
        class TrainingConfig:
            epochs: int = 10
            batch_size: int = 32
            grad_accum_steps: int = 1
            mixed_precision: bool = False
            log_every: int = 100
            eval_every: int = 1000
            tags: List[str] = field(default_factory=list)

        @dataclass
        class AppConfig:
            # Schema for the whole final config tree.
            db: DBConfig = field(default_factory=DBConfig)
            model: ModelConfig = field(default_factory=ModelConfig)
            optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
            training: TrainingConfig = field(default_factory=TrainingConfig)

            experiment_name: str = MISSING
            seed: int = 42
            debug: bool = False
            output_dir: str = "/tmp/outputs"

            # Extra strict research/production guard:
            # Hydra's +key=value can intentionally append unknown keys.
            # This flag lets app.py reject those extras after composition.
            strict_no_extra_keys: bool = True
    """,
    )

    write(
        '04_structured_configs/conf/config.yaml',
        """
        # Critical: this activates the dataclass schema.
        defaults:
          - app_schema
          - _self_

        experiment_name: structured_example
        seed: 123
        debug: false

        db:
          name: tutorial_db
          password: "hunter2"
          host: db.example.com

        model:
          hidden_size: 512
          num_layers: 8

        optimizer:
          lr: 3.0e-4
          weight_decay: 1.0e-2
          clip_grad_norm: 1.0

        training:
          epochs: 20
          batch_size: 64
          mixed_precision: true
          tags: [experiment, structured]
    """,
    )

    write(
        '04_structured_configs/conf/config_missing.yaml',
        """
        # Intentionally missing required fields.
        defaults:
          - app_schema
          - _self_

        seed: 123
        debug: false

        # Missing:
        #   experiment_name
        #   db.name
        #   db.password
    """,
    )

    write(
        '04_structured_configs/conf/config_plain.yaml',
        """
        # Plain YAML version: no schema in defaults.
        experiment_name: plain_yaml_example
        seed: 123
        db:
          name: tutorial_db
          password: hunter2
        training:
          epochs: 20
    """,
    )

    write(
        '04_structured_configs/app.py',
        """
        \"\"\"
        LESSON 4 — Structured configs + ConfigStore.

        Flow:
          1. schemas.py defines AppConfig, the rulebook for the whole tree.
          2. ConfigStore registers that rulebook as "app_schema".
          3. conf/config.yaml includes `- app_schema` in defaults.
          4. Hydra starts with dataclass defaults and merges YAML on top.
          5. Types are checked. MISSING fields are forced at startup below.
        \"\"\"
        import hydra
        from dataclasses import fields, is_dataclass
        from hydra.core.config_store import ConfigStore
        from omegaconf import DictConfig, OmegaConf
        from schemas import AppConfig

        cs = ConfigStore.instance()
        cs.store(name="app_schema", node=AppConfig)

        def reject_unknown_keys(node: DictConfig, schema_type: type, prefix: str = "") -> None:
            '''Optional extra guard: reject keys added with +key=value or ++key=value.

            Hydra's built-in structured config validation catches:
              training.epochs=not_an_int
              training.nonexistent=1

            But Hydra's CLI append operators are intentionally allowed to add:
              +training.nonexistent=1
              ++training.nonexistent=1

            This function makes the app stricter by comparing the final config
            against the dataclass fields after Hydra has composed it.
            '''
            if not is_dataclass(schema_type) or not isinstance(node, DictConfig):
                return

            allowed = {f.name: f.type for f in fields(schema_type)}
            for key in node.keys():
                if key not in allowed:
                    raise KeyError(f"Unknown config key added by override: {prefix}{key}")

                child_schema = allowed[key]
                child_node = node.get(key)
                if is_dataclass(child_schema) and isinstance(child_node, DictConfig):
                    reject_unknown_keys(child_node, child_schema, prefix=f"{prefix}{key}.")

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig) -> None:
            # The strict unknown-key guard is an optional app-level addition.
            # It is enabled by the schema config, but absent in config_plain.yaml.
            if bool(cfg.get("strict_no_extra_keys", False)):
                reject_unknown_keys(cfg, AppConfig)

            # Force MISSING/interpolations to fail immediately instead of later.
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

            print("\\n=== Structured Config Demo ===\\n")
            print(OmegaConf.to_yaml(cfg))

            print("--- Schema info ---")
            print(f"OmegaConf type of root : {OmegaConf.get_type(cfg)}")
            if "training" in cfg:
                print(f"OmegaConf type training: {OmegaConf.get_type(cfg.training)}")

            print("\\n--- Runtime value types ---")
            if "training" in cfg and "epochs" in cfg.training:
                print(f"epochs type    : {type(cfg.training.epochs)}")
            if "optimizer" in cfg and "lr" in cfg.optimizer:
                print(f"lr type        : {type(cfg.optimizer.lr)}")
            if "training" in cfg and "mixed_precision" in cfg.training:
                print(f"mixed_prec type: {type(cfg.training.mixed_precision)}")
            if "training" in cfg and "tags" in cfg.training:
                print(f"tags type      : {type(cfg.training.tags)}")
        if __name__ == "__main__":
            main()
    """,
    )

    write(
        '04_structured_configs/LESSON.md',
        r"""
        # Lesson 4 — Structured Configs

        Structured configs are optional. They can be overkill for fast PhD
        research code, but they are useful when config mistakes waste time.

        ## Main idea

        ```text
        dataclass schema       = allowed structure + types + default values
        ConfigStore.instance() = register schema with Hydra
        defaults: - app_schema = actually use the schema
        YAML values            = override schema defaults
        ```

        ## What `AppConfig` does

        `AppConfig` describes the whole final config tree:

        ```python
        @dataclass
        class AppConfig:
            db: DBConfig
            model: ModelConfig
            optimizer: OptimizerConfig
            training: TrainingConfig
            experiment_name: str
            seed: int
        ```

        It tells Hydra that `training.epochs` must be an int, `optimizer.lr`
        must be a float, ordinary unknown overrides should fail.

        ## Why add it to `config.yaml`?

        This registers the schema:

        ```python
        cs.store(name="app_schema", node=AppConfig)
        ```

        But registration alone does not activate it.

        This activates it:

        ```yaml
        defaults:
          - app_schema
          - _self_
        ```

        It means: start with AppConfig defaults, then merge this YAML file on top.

        Without `- app_schema`, Hydra treats the file as plain YAML, so
        `training.epochs=not_an_int` can become a string.

        ## Run

        ```bash
        cd 04_structured_configs
        python app.py
        ```

        ## Type mismatch

        ```bash
        python app.py training.epochs=50
        python app.py training.epochs=not_an_int
        ```

        The second command should fail when the schema is active.

        ## MISSING fields

        In the schema:

        ```python
        experiment_name: str = MISSING
        ```

        means the value must be provided by YAML or CLI.

        This intentionally fails:

        ```bash
        python app.py --config-name config_missing
        ```

        ## Unknown keys

        Built-in structured config validation catches ordinary unknown-key
        overrides:

        ```bash
        python app.py training.nonexistent=1
        ```

        Important nuance: Hydra's `+` and `++` operators are designed to append
        or upsert keys, so they can intentionally add keys even when a schema is
        active.

        This tutorial adds an optional Python-side strict validator:

        ```python
        strict_no_extra_keys: true
        ```

        Therefore this also fails in this lesson:

        ```bash
        python app.py +training.nonexistent=1
        ```

        To see Hydra's default append behavior, disable the extra guard:

        ```bash
        python app.py strict_no_extra_keys=false +training.nonexistent=1
        ```

        Check schema activation with:

        ```bash
        python app.py --info defaults
        ```

        You should see `app_schema` loaded before `config`.

        ## Plain YAML comparison

        This intentionally does not activate the schema:

        ```bash
        python app.py --config-name config_plain training.epochs=not_an_int
        ```

        In plain YAML mode, Hydra does not know that epochs should be int.

        ## Recommendation

        For most research code, start with YAML-only Hydra plus manual sanity
        checks. Add structured configs later for stable sections like training,
        optimizer, checkpointing, and logging.
    """,
    )

    note('Created: 04_structured_configs/ with working schema activation')


def create_05_instantiate() -> None:
    """Create the object-instantiation lesson."""
    section('05_instantiate — Building Python objects from YAML')

    write(
        '05_instantiate/conf/config.yaml',
        """
        defaults:
          - optimizer: adam
          - scheduler: cosine
          - _self_

        # Instantiate an object directly from YAML using _target_
        # hydra.utils.instantiate(cfg.dataset) will call:
        #   FakeDataset(path='/data/train', split='train', augment=True)
        dataset:
          _target_: components.FakeDataset
          path: /data/train
          split: train
          augment: true

        model:
          _target_: components.SimpleModel
          input_size: 784
          hidden_size: 256
          output_size: 10
          dropout: 0.2
    """,
    )

    write(
        '05_instantiate/conf/optimizer/adam.yaml',
        """
        # @package _global_
        optimizer:
          _target_: components.FakeAdam
          lr: 0.001
          betas: [0.9, 0.999]
          weight_decay: 0.0
    """,
    )

    write(
        '05_instantiate/conf/optimizer/sgd.yaml',
        """
        # @package _global_
        optimizer:
          _target_: components.FakeSGD
          lr: 0.01
          momentum: 0.9
          nesterov: true
    """,
    )

    write(
        '05_instantiate/conf/scheduler/cosine.yaml',
        """
        # @package _global_
        scheduler:
          _target_: components.FakeCosineScheduler
          T_max: 100
          eta_min: 1.0e-6
    """,
    )

    write(
        '05_instantiate/conf/scheduler/step.yaml',
        """
        # @package _global_
        scheduler:
          _target_: components.FakeStepScheduler
          step_size: 30
          gamma: 0.1
    """,
    )

    write(
        '05_instantiate/components.py',
        """
        \"\"\"
        Fake components to demonstrate instantiate().
        In a real project these would be torch.nn.Module, torch.optim, etc.
        \"\"\"

        class FakeDataset:
            def __init__(self, path, split, augment=False):
                self.path = path
                self.split = split
                self.augment = augment
            def __repr__(self):
                return f"FakeDataset(path={self.path!r}, split={self.split!r}, augment={self.augment})"

        class SimpleModel:
            def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.dropout = dropout
            def __repr__(self):
                return (f"SimpleModel({self.input_size}→{self.hidden_size}→"
                        f"{self.output_size}, dropout={self.dropout})")

        class FakeAdam:
            def __init__(self, lr, betas, weight_decay=0.0):
                self.lr = lr
                self.betas = betas
                self.weight_decay = weight_decay
            def __repr__(self):
                return f"Adam(lr={self.lr}, betas={self.betas})"

        class FakeSGD:
            def __init__(self, lr, momentum=0.0, nesterov=False):
                self.lr = lr
                self.momentum = momentum
                self.nesterov = nesterov
            def __repr__(self):
                return f"SGD(lr={self.lr}, momentum={self.momentum})"

        class FakeCosineScheduler:
            def __init__(self, T_max, eta_min=0.0):
                self.T_max = T_max
                self.eta_min = eta_min
            def __repr__(self):
                return f"CosineScheduler(T_max={self.T_max}, eta_min={self.eta_min})"

        class FakeStepScheduler:
            def __init__(self, step_size, gamma=0.1):
                self.step_size = step_size
                self.gamma = gamma
            def __repr__(self):
                return f"StepScheduler(step_size={self.step_size}, gamma={self.gamma})"
    """,
    )

    write(
        '05_instantiate/app.py',
        """
        \"\"\"
        LESSON 5 — hydra.utils.instantiate()

        _target_ in YAML tells Hydra which Python class/function to call.
        The rest of the keys become keyword arguments.

        instantiate() modes:
          _recursive_=True  (default) — recursively instantiate nested _target_ objects
          _recursive_=False — only instantiate the top level
          _convert_="none"  (default) — pass DictConfig/ListConfig as-is
          _convert_="all"   — convert DictConfig→dict, ListConfig→list before calling
          _convert_="partial" — convert only non-structured configs

        instantiate() vs call():
          instantiate(cfg)     → calls cfg._target_(**rest_of_keys)
          call(cfg)            → alias for instantiate (same thing)
        \"\"\"
        import hydra
        from hydra.utils import instantiate
        from omegaconf import DictConfig, OmegaConf

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig) -> None:
            print("\\n=== instantiate() Demo ===\\n")
            print("Config:")
            print(OmegaConf.to_yaml(cfg))

            # ── Instantiate each component ─────────────────────────────────
            dataset   = instantiate(cfg.dataset)
            model     = instantiate(cfg.model)
            optimizer = instantiate(cfg.optimizer)
            scheduler = instantiate(cfg.scheduler)

            print("Built objects:")
            print(f"  Dataset   : {dataset}")
            print(f"  Model     : {model}")
            print(f"  Optimizer : {optimizer}")
            print(f"  Scheduler : {scheduler}")

            # ── Override constructor args at instantiation time ────────────
            # Pass extra kwargs — they override what's in the config
            val_dataset = instantiate(cfg.dataset, split="val", augment=False)
            print(f"\\n  Val dataset: {val_dataset}")

            # ── Partial instantiation ──────────────────────────────────────
            # Use _partial_=True to get a functools.partial instead of
            # the object — useful when you need to pass args later
            # (e.g., pass model.parameters() to optimizer)
            opt_partial = instantiate(cfg.optimizer, _partial_=True)
            print(f"\\n  Partial optimizer type: {type(opt_partial)}")
            # In real code: optimizer = opt_partial(params=model.parameters())

        if __name__ == "__main__":
            main()
    """,
    )

    write(
        '05_instantiate/LESSON.md',
        r"""
        # Lesson 5 — instantiate()

        ## Default run (Adam optimizer + Cosine scheduler)
        ```
        cd 05_instantiate
        python app.py
        ```

        ## Swap optimizer — the class that gets built changes
        ```
        python app.py optimizer=sgd
        ```

        ## Swap scheduler
        ```
        python app.py scheduler=step
        ```

        ## Override a constructor arg
        ```
        python app.py optimizer.lr=1e-5
        python app.py model.hidden_size=512 model.dropout=0.3
        ```

        ## Key: _target_ format
        `_target_` must be a fully-qualified Python dotted path:
        ```yaml
        _target_: mypackage.submodule.MyClass
        _target_: torch.optim.Adam           # from torch
        _target_: torch.nn.CrossEntropyLoss  # etc.
        ```

        ## _recursive_ (default True)
        If your config has nested objects that also have _target_,
        Hydra instantiates them all automatically (depth-first).

        ## _partial_=True
        Returns a functools.partial — call it later with remaining args.
        Perfect for optimizers: define lr/weight_decay in config,
        pass model.parameters() in code.

        ```python
        opt_fn = instantiate(cfg.optimizer, _partial_=True)
        optimizer = opt_fn(params=model.parameters())
        ```

        ## _convert_="all"
        Converts OmegaConf containers to plain Python dicts/lists
        before passing to the constructor. Use when your class expects
        plain Python types, not OmegaConf wrappers.

        ## Registry vs Hydra instantiate

        You do not need to use `_target_` everywhere. For research ML templates,
        a Registry is often easier to debug for models, datasets, losses, and
        metrics:

        ```yaml
        model:
          name: resnet50
        optimizer:
          name: adamw
          lr: 0.0003
        ```

        ```python
        model = MODEL_REGISTRY.build(cfg.model.name, cfg.model)
        optimizer = OPTIMIZER_REGISTRY.build(
            cfg.optimizer.name,
            model.parameters(),
            cfg.optimizer,
        )
        ```

        Good practical split:

        ```text
        Registry:      models, datasets, losses, metrics, trainers
        instantiate(): optional; useful for stable third-party constructors
        ```
    """,
    )

    note('Created: 05_instantiate/ with _target_ and _partial_ demos')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — MULTIRUN
# ─────────────────────────────────────────────────────────────────────────────


def create_06_multirun() -> None:
    """Create the multirun and sweep lesson."""
    section('06_multirun — Sweeping over config combinations')

    write(
        '06_multirun/conf/config.yaml',
        """
        defaults:
          - model: small
          - optimizer: adam
          - _self_

        experiment_name: sweep
        seed: 42
    """,
    )

    write(
        '06_multirun/conf/model/small.yaml',
        """
        # @package _global_
        model:
          name: small
          hidden_size: 128
          num_layers: 2
    """,
    )

    write(
        '06_multirun/conf/model/large.yaml',
        """
        # @package _global_
        model:
          name: large
          hidden_size: 512
          num_layers: 8
    """,
    )

    write(
        '06_multirun/conf/optimizer/adam.yaml',
        """
        # @package _global_
        optimizer:
          name: adam
          lr: 0.001
    """,
    )

    write(
        '06_multirun/conf/optimizer/sgd.yaml',
        """
        # @package _global_
        optimizer:
          name: sgd
          lr: 0.01
    """,
    )

    write(
        '06_multirun/train.py',
        """
        \"\"\"
        LESSON 6 — Multirun.

        -m / --multirun enables sweep mode.
        It is not needed for normal single-run overrides.
        \"\"\"
        import os
        import random
        import hydra
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import DictConfig

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def train(cfg: DictConfig) -> None:
            hcfg = HydraConfig.get()

            job_num = hcfg.job.get("num", 0)
            overrides = list(hcfg.overrides.task)
            output_dir = hcfg.runtime.output_dir

            random.seed(cfg.seed + int(job_num))
            fake_loss = round(random.uniform(0.1, 2.0), 4)
            fake_acc  = round(random.uniform(0.6, 0.99), 4)

            print(f"\\n[Run {job_num}] {cfg.model.name} + {cfg.optimizer.name}  lr={cfg.optimizer.lr}")
            print(f"  overrides : {overrides}")
            print(f"  output_dir: {output_dir}")
            print(f"  cwd       : {os.getcwd()}")
            print(f"  loss={fake_loss}  acc={fake_acc}")

        if __name__ == "__main__":
            train()
    """,
    )

    write(
        '06_multirun/LESSON.md',
        r"""
        # Lesson 6 — Multirun

        Run all commands from `06_multirun/`.

        ## Normal runs do not need `-m`

        ```bash
        python train.py
        python train.py model=large
        python train.py optimizer.lr=0.0003
        python train.py model=large optimizer=sgd
        ```

        ## Use `-m` for sweeps

        ```bash
        python train.py -m optimizer.lr=0.1,0.01,0.001
        ```

        This launches 3 jobs.

        ## Cartesian product

        ```bash
        python train.py -m model=small,large optimizer=adam,sgd
        ```

        This launches 2 x 2 = 4 jobs.

        ## Groups + values

        ```bash
        python train.py -m model=small,large optimizer.lr=0.1,0.01,0.001
        ```

        This launches 2 x 3 = 6 jobs.

        ## Range syntax

        ```bash
        python train.py -m 'optimizer.lr=range(0.001,0.1,0.01)'
        ```

        Values like `0.011000000000000001` are normal floating-point behavior.

        ## Glob syntax

        ```bash
        python train.py -m 'model=glob(*)'
        ```

        This sweeps every config file in `conf/model/`. If the folder only has
        `small.yaml` and `large.yaml`, it launches 2 jobs.

        ## Hydra logging

        These lines are Hydra's default logging, not code you wrote:

        ```text
        [HYDRA] Launching 4 jobs locally
        [HYDRA]        #0 : model=small optimizer=adam
        ```

        Your script's output starts at `[Run ...]`.

        Hydra logging comes from internal configs such as:

        ```text
        hydra/hydra_logging/default
        hydra/job_logging/default
        ```

        You can see them with:

        ```bash
        python train.py --info defaults
        ```

        ## Parallel multirun

        ```bash
        python train.py -m hydra/launcher=joblib \
            hydra.launcher.n_jobs=4 \
            model=small,large optimizer=adam,sgd
        ```

        ## Access current run info

        ```python
        from hydra.core.hydra_config import HydraConfig
        hcfg = HydraConfig.get()
        hcfg.job.num
        hcfg.overrides.task
        hcfg.runtime.output_dir
        ```
    """,
    )

    note('Created: 06_multirun/ with corrected -m and logging explanation')


def create_07_plugins() -> None:
    """Create the Hydra plugins lesson."""
    section('07_plugins — Advanced patterns')

    write(
        '07_plugins/conf/config.yaml',
        """
        defaults:
          - _self_
          - override hydra/output: custom   # override hydra's own output config

        project: advanced_demo
        data_root: /data

        # Packages: control where a config group's keys land in the tree
        # See conf/extra/ for examples
    """,
    )

    write(
        '07_plugins/conf/hydra/output/custom.yaml',
        """
        # Override Hydra's default output dir structure.
        # This config goes under hydra/output/ and overrides hydra's built-in.
        run:
          dir: outputs/${now:%Y-%m-%d_%H-%M-%S}_${hydra.job.name}
        sweep:
          dir: sweeps/${now:%Y-%m-%d}
          subdir: ${hydra.job.num}
    """,
    )

    write(
        '07_plugins/conf/extra/logging.yaml',
        """
        # @package _global_
        # Using @package _global_ puts keys at the top level,
        # not nested under 'extra.logging'
        logging:
          level: INFO
          format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
          file: ${project}_run.log
    """,
    )

    write(
        '07_plugins/conf/extra/callbacks.yaml',
        """
        # @package _global_
        callbacks:
          early_stopping:
            monitor: val_loss
            patience: 5
            min_delta: 0.001
          checkpoint:
            save_top_k: 3
            monitor: val_acc
            mode: max
    """,
    )

    write(
        '07_plugins/app.py',
        """
        \"\"\"
        LESSON 7 — Advanced patterns:

        1. @package directive — control where config keys land in the tree
        2. Overriding Hydra's internal config (output dirs, logging, launcher)
        3. Config composition order and _self_
        4. hydra.utils.get_original_cwd() — get the CWD before Hydra changed it
        5. hydra.utils.to_absolute_path() — resolve paths relative to original CWD
        6. Callbacks pattern — append optional config blocks
        \"\"\"
        import hydra
        from hydra.utils import get_original_cwd, to_absolute_path
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import DictConfig, OmegaConf
        import os

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig) -> None:
            hcfg = HydraConfig.get()

            print("\\n=== Advanced Hydra Patterns ===\\n")
            print(OmegaConf.to_yaml(cfg))

            # ── Path helpers ───────────────────────────────────────────────
            original_cwd = get_original_cwd()
            print(f"Original CWD  : {original_cwd}")
            print(f"Current CWD   : {os.getcwd()}")   # Hydra changed this!

            # to_absolute_path resolves relative to original CWD
            abs_data = to_absolute_path("data/train.csv")
            print(f"Abs data path : {abs_data}")

            # ── HydraConfig internals ──────────────────────────────────────
            print(f"\\nJob name      : {hcfg.job.name}")
            print(f"Config sources: {[s.path for s in hcfg.runtime.config_sources]}")

        if __name__ == "__main__":
            main()
    """,
    )

    write(
        '07_plugins/advanced_patterns.py',
        """
        \"\"\"
        Advanced pattern: config groups with @package directive.

        @package controls the namespace where a config is merged.
        Examples:
          # @package _global_          → merge at root level
          # @package model.encoder     → merge under model.encoder
          # @package _group_           → merge under the group name (default)
          # (no directive)             → same as _group_

        This file shows how to programmatically compose configs
        without the @hydra.main decorator — useful for unit tests
        and scripts that need Hydra config but aren't entry points.
        \"\"\"
        from hydra import compose, initialize, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        def demo_compose_api():
            \"\"\"
            The compose API lets you use Hydra config loading
            in any Python script — no decorator needed.
            Great for unit tests!
            \"\"\"
            # Clear any existing Hydra state
            GlobalHydra.instance().clear()

            # initialize points to the config dir (relative to this file)
            with initialize(config_path="conf", version_base=None):
                # compose builds the merged config
                cfg = compose(config_name="config")
                print("\\n=== compose() API ===")
                print(OmegaConf.to_yaml(cfg))

                # compose with overrides
                cfg2 = compose(
                    config_name="config",
                    overrides=["project=test_project", "+logging.level=DEBUG"]
                )
                print("\\n=== compose() with overrides ===")
                print(f"project: {cfg2.project}")

        if __name__ == "__main__":
            demo_compose_api()
    """,
    )

    write(
        '07_plugins/LESSON.md',
        r"""
        # Lesson 7 — Advanced Patterns

        ## Run the main app
        ```
        cd 07_plugins
        python app.py
        ```

        ## Add optional config blocks (append groups)
        ```
        python app.py '+extra=[logging,callbacks]'
        ```

        ## Use the compose API (no decorator — great for tests)
        ```
        python advanced_patterns.py
        ```

        ## @package directive quick reference
        In any YAML file, the first line can be a package directive:
        ```yaml
        # @package _global_       ← merge at root of config tree
        # @package model.encoder  ← merge under model.encoder
        # @package _group_        ← merge under the group name (default)
        ```
        Without a directive, Hydra uses the group name as the package.

        ## get_original_cwd() — critical for file paths!
        Hydra CHANGES the working directory to the run output dir.
        This breaks relative file paths. Use:
        ```python
        from hydra.utils import get_original_cwd, to_absolute_path
        data_path = to_absolute_path(cfg.data_root)  # always correct
        ```

        ## Overriding Hydra's own config
        Hydra has an internal config tree (hydra.*) you can override:
        ```
        python app.py hydra.run.dir=/my/output
        python app.py hydra.job.name=my_job
        python app.py hydra.verbose=true          # verbose logging
        python app.py 'hydra.output_subdir=null'  # disable .hydra/ subdir
        ```

        ## Disabling config output dir
        ```
        python app.py hydra.output_subdir=null hydra.run.dir=.
        ```
        This makes Hydra NOT change the CWD and NOT create output dirs.
        Useful for interactive scripts.
    """,
    )

    note('Created: 07_plugins/ with @package, compose API, and path helpers')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — REAL WORLD ML PROJECT
# ─────────────────────────────────────────────────────────────────────────────


def create_08_real_world() -> None:
    """Create the realistic ML project lesson."""
    section('08_real_world_ml — Production-style ML project')

    write(
        '08_real_world_ml/conf/config.yaml',
        """
        defaults:
          - dataset: mnist
          - model: cnn
          - optimizer: adamw
          - scheduler: cosine
          - callbacks: default
          - logger: console
          - training       # loads conf/training.yaml
          - _self_

        # ── Experiment metadata ────────────────────────────────────────────
        experiment_name: ???         # REQUIRED: set from CLI
        run_id: ${now:%Y%m%d_%H%M%S}
        seed: 42
        debug: false

        # ── Paths ─────────────────────────────────────────────────────────
        data_dir: /data
        output_dir: /outputs/${experiment_name}/${run_id}

        # ── Hydra output control ───────────────────────────────────────────
        hydra:
          run:
            dir: ${output_dir}
          sweep:
            dir: sweeps/${experiment_name}
            subdir: ${hydra.job.num}
    """,
    )

    write(
        '08_real_world_ml/conf/dataset/mnist.yaml',
        """
        # @package _global_
        dataset:
          _target_: project.data.FakeDataset
          name: mnist
          data_dir: ${data_dir}/mnist
          image_size: 28
          num_classes: 10
          train_split: 0.9
          num_workers: 4
          pin_memory: true
    """,
    )

    write(
        '08_real_world_ml/conf/dataset/cifar10.yaml',
        """
        # @package _global_
        dataset:
          _target_: project.data.FakeDataset
          name: cifar10
          data_dir: ${data_dir}/cifar10
          image_size: 32
          num_classes: 10
          train_split: 0.9
          num_workers: 8
          pin_memory: true
    """,
    )

    write(
        '08_real_world_ml/conf/model/cnn.yaml',
        """
        # @package _global_
        model:
          _target_: project.models.FakeCNN
          name: cnn_small
          in_channels: 1
          num_classes: ${dataset.num_classes}   # interpolate from dataset!
          hidden_channels: [32, 64, 128]
          dropout: 0.3
    """,
    )

    write(
        '08_real_world_ml/conf/model/resnet.yaml',
        """
        # @package _global_
        model:
          _target_: project.models.FakeResNet
          name: resnet18
          in_channels: 3
          num_classes: ${dataset.num_classes}
          pretrained: false
          dropout: 0.1
    """,
    )

    write(
        '08_real_world_ml/conf/optimizer/adamw.yaml',
        """
        # @package _global_
        optimizer:
          _target_: project.optimizers.FakeAdamW
          lr: 3.0e-4
          betas: [0.9, 0.95]
          weight_decay: 0.05
          eps: 1.0e-8
    """,
    )

    write(
        '08_real_world_ml/conf/optimizer/sgd.yaml',
        """
        # @package _global_
        optimizer:
          _target_: project.optimizers.FakeSGD
          lr: 0.05
          momentum: 0.9
          weight_decay: 5.0e-4
          nesterov: true
    """,
    )

    write(
        '08_real_world_ml/conf/scheduler/cosine.yaml',
        """
        # @package _global_
        scheduler:
          _target_: project.schedulers.FakeCosineScheduler
          T_max: ${training.epochs}   # interpolate from training config!
          eta_min: 1.0e-6
          warmup_epochs: 5
    """,
    )

    write(
        '08_real_world_ml/conf/scheduler/onecycle.yaml',
        """
        # @package _global_
        scheduler:
          _target_: project.schedulers.FakeOneCycleLR
          max_lr: ${optimizer.lr}
          epochs: ${training.epochs}
          pct_start: 0.3
    """,
    )

    write(
        '08_real_world_ml/conf/callbacks/default.yaml',
        """
        # @package _global_
        callbacks:
          early_stopping:
            monitor: val/loss
            patience: 10
            min_delta: 1.0e-4
            mode: min
          model_checkpoint:
            dirpath: ${output_dir}/checkpoints
            filename: epoch={epoch}-val_acc={val/acc:.3f}
            monitor: val/acc
            mode: max
            save_top_k: 3
            save_last: true
          lr_monitor:
            logging_interval: step
    """,
    )

    write(
        '08_real_world_ml/conf/logger/console.yaml',
        """
        # @package _global_
        logger:
          _target_: project.loggers.ConsoleLogger
          level: INFO
          log_every_n_steps: 10
    """,
    )

    write(
        '08_real_world_ml/conf/logger/wandb.yaml',
        """
        # @package _global_
        logger:
          _target_: project.loggers.FakeWandbLogger
          project: ${experiment_name}
          name: ${run_id}
          tags: [hydra, tutorial]
          log_model: true
    """,
    )

    write(
        '08_real_world_ml/conf/training.yaml',
        """
        # @package _global_
        training:
          epochs: 50
          batch_size: 128
          grad_accum_steps: 1
          mixed_precision: false
          grad_clip_norm: 1.0
          val_check_interval: 1.0
          limit_train_batches: 1.0
          limit_val_batches: 1.0
    """,
    )

    write('08_real_world_ml/project/__init__.py', '')

    write(
        '08_real_world_ml/project/data.py',
        """
        class FakeDataset:
            def __init__(self, name, data_dir, image_size, num_classes,
                         train_split=0.9, num_workers=4, pin_memory=True):
                self.name = name
                self.data_dir = data_dir
                self.image_size = image_size
                self.num_classes = num_classes
            def __repr__(self):
                return f"Dataset({self.name}, {self.image_size}x{self.image_size}, classes={self.num_classes})"
    """,
    )

    write(
        '08_real_world_ml/project/models.py',
        """
        class FakeCNN:
            def __init__(self, name, in_channels, num_classes, hidden_channels, dropout=0.0):
                self.name = name
                self.in_channels = in_channels
                self.num_classes = num_classes
                self.hidden_channels = list(hidden_channels)
            def __repr__(self):
                return f"CNN({self.name}, channels={self.hidden_channels}, out={self.num_classes})"

        class FakeResNet:
            def __init__(self, name, in_channels, num_classes, pretrained=False, dropout=0.0):
                self.name = name
                self.num_classes = num_classes
                self.pretrained = pretrained
            def __repr__(self):
                return f"ResNet({self.name}, pretrained={self.pretrained}, out={self.num_classes})"
    """,
    )

    write(
        '08_real_world_ml/project/optimizers.py',
        """
        class FakeAdamW:
            def __init__(self, lr, betas, weight_decay=0.0, eps=1e-8):
                self.lr = lr
                self.betas = list(betas)
                self.weight_decay = weight_decay
            def __repr__(self):
                return f"AdamW(lr={self.lr}, wd={self.weight_decay})"

        class FakeSGD:
            def __init__(self, lr, momentum=0.0, weight_decay=0.0, nesterov=False):
                self.lr = lr
                self.momentum = momentum
            def __repr__(self):
                return f"SGD(lr={self.lr}, momentum={self.momentum})"
    """,
    )

    write(
        '08_real_world_ml/project/schedulers.py',
        """
        class FakeCosineScheduler:
            def __init__(self, T_max, eta_min=0.0, warmup_epochs=0):
                self.T_max = T_max
                self.eta_min = eta_min
                self.warmup_epochs = warmup_epochs
            def __repr__(self):
                return f"CosineScheduler(T_max={self.T_max}, warmup={self.warmup_epochs})"

        class FakeOneCycleLR:
            def __init__(self, max_lr, epochs, pct_start=0.3):
                self.max_lr = max_lr
                self.epochs = epochs
            def __repr__(self):
                return f"OneCycleLR(max_lr={self.max_lr}, epochs={self.epochs})"
    """,
    )

    write(
        '08_real_world_ml/project/loggers.py',
        """
        class ConsoleLogger:
            def __init__(self, level="INFO", log_every_n_steps=10):
                self.level = level
                self.log_every_n_steps = log_every_n_steps
            def __repr__(self):
                return f"ConsoleLogger(level={self.level})"

        class FakeWandbLogger:
            def __init__(self, project, name, tags=None, log_model=False):
                self.project = project
                self.name = name
                self.tags = list(tags) if tags else []
            def __repr__(self):
                return f"WandbLogger(project={self.project}, run={self.name})"
    """,
    )

    write(
        '08_real_world_ml/train.py',
        """
        \"\"\"
        LESSON 8 — Production-style ML project with Hydra.

        This demonstrates the full pattern used in real ML codebases:
          • All components defined with _target_ + instantiate()
          • Cross-config interpolation (model.num_classes from dataset)
          • Structured experiment naming (experiment_name required)
          • Custom output dirs per run
          • Optional logger swap (console vs wandb)
        \"\"\"
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))

        import hydra
        from hydra.utils import instantiate
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import DictConfig, OmegaConf

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def train(cfg: DictConfig) -> None:
            hcfg = HydraConfig.get()

            print("\\n" + "="*60)
            print(f"  Experiment : {cfg.experiment_name}  [{cfg.run_id}]")
            print(f"  Seed       : {cfg.seed}")
            print("="*60)

            # ── Instantiate all components ─────────────────────────────────
            dataset   = instantiate(cfg.dataset)
            model     = instantiate(cfg.model)
            optimizer = instantiate(cfg.optimizer)
            scheduler = instantiate(cfg.scheduler)
            logger    = instantiate(cfg.logger)

            print(f"\\n  Dataset   : {dataset}")
            print(f"  Model     : {model}")
            print(f"  Optimizer : {optimizer}")
            print(f"  Scheduler : {scheduler}")
            print(f"  Logger    : {logger}")

            print(f"\\n  Output dir: {cfg.output_dir}")
            print(f"  Run dir   : {hcfg.run.dir}")

            print("\\n--- Training config ---")
            print(OmegaConf.to_yaml(cfg))

        if __name__ == "__main__":
            train()
    """,
    )

    write(
        '08_real_world_ml/LESSON.md',
        r"""
        # Lesson 8 — Real-World ML Project

        All commands from 08_real_world_ml/.

        ## Basic run (experiment_name is required — use ??? in YAML)
        ```
        python train.py experiment_name=baseline
        ```

        ## Swap dataset + model
        ```
        python train.py experiment_name=cifar_run dataset=cifar10 model=resnet
        ```

        ## Swap logger (e.g. to WandB)
        ```
        python train.py experiment_name=wandb_test logger=wandb
        ```

        ## Full sweep: dataset x model x optimizer (8 runs)
        ```
        python train.py -m \
            experiment_name=big_sweep \
            dataset=mnist,cifar10 \
            model=cnn,resnet \
            optimizer=adamw,sgd
        ```

        ## Cross-config interpolation
        Notice in conf/model/cnn.yaml:
            num_classes: ${dataset.num_classes}
        Hydra resolves this from the currently-loaded dataset config!
        Switch dataset and num_classes updates automatically.

        ## Override scheduler T_max (which interpolates from training.epochs)
        ```
        python train.py experiment_name=long training.epochs=200
        ```
        The scheduler's T_max automatically updates to 200.

        ## The ??? pattern (required fields)
        experiment_name: ???  means Hydra errors if not provided.
        This prevents accidentally running with a missing name.
        ```
        python train.py  # ← ERROR: experiment_name is missing
        python train.py experiment_name=my_run  # ← OK
        ```
    """,
    )

    note('Created: 08_real_world_ml/ — full ML project with all patterns combined')


# ─────────────────────────────────────────────────────────────────────────────
# README
# ─────────────────────────────────────────────────────────────────────────────


def create_readme() -> None:
    """Create the generated tutorial repository README."""
    write(
        'README.md',
        """
        # Hydra Config Framework — Complete Tutorial Repository

        Generated by `hydra_learning_setup.py`. Work through lessons in order.

        ## Prerequisites
        ```bash
        pip install hydra-core omegaconf hydra-joblib-launcher
        ```

        ## Lesson Map

        | # | Folder | Concepts |
        |---|--------|----------|
        | 0 | 00_basics | @hydra.main, DictConfig, dot access, --cfg |
        | 1 | 01_config_groups | defaults list, config groups, @package, _self_ |
        | 2 | 02_overrides | CLI overrides, shell quoting, + vs ++, deletion |
        | 3 | 03_interpolation | ${ref}, oc.env, oc.select, custom resolvers, lazy eval |
        | 4 | 04_structured_configs | dataclasses, ConfigStore, active schema, MISSING |
        | 5 | 05_instantiate | _target_, _partial_, _recursive_, _convert_, Registry comparison |
        | 6 | 06_multirun | -m, sweeps, glob/range, Hydra logging |
        | 7 | 07_plugins | compose API, path helpers, hydra.* |
        | 8 | 08_real_world_ml | Full ML project — all patterns combined |

        ## Override operators

        | Operator | If key exists | If key does not exist | Meaning |
        |----------|---------------|-----------------------|---------|
        | `key=value` | Updates it | Error | Modify existing key only |
        | `+key=value` | Error | Adds it | Insert new key only |
        | `++key=value` | Updates it | Adds it | Upsert: update or insert |
        | `~key` | Deletes it | Error | Remove a key |
        | `key=null` | Sets to None | Error | Null assignment |

        ## Shell quoting equivalences

        ```bash
        python demo.py 'app.tags=[staging,v2,hotfix]'
        python demo.py app.tags="[staging,v2,hotfix]"

        python demo.py 'app.name="hello:world"'
        python demo.py app.name="hello:world"
        python demo.py 'app.name=hello:world'

        python demo.py '+server={host:127.0.0.1,port:9090,ssl:true}'
        python demo.py +server="{host:127.0.0.1,port:9090,ssl:true}"
        ```

        ## Useful CLI flags

        ```bash
        --cfg job        # print composed application config, don't run app body
        --cfg hydra      # print Hydra's internal config
        --cfg all        # print both app config and Hydra config
        --info defaults  # show which config files were loaded
        -m / --multirun  # sweep mode; needed only for multiple generated jobs
        --help           # show config help
        ```

        ## Practical recommendation for research ML

        Start with:

        ```text
        Hydra YAML config groups + CLI overrides + interpolation + multirun
        ```

        Add structured configs only when the schema is stable or config mistakes
        are wasting GPU time.

        Registry is often easier than Hydra `_target_` for models/datasets:

        ```python
        model = MODEL_REGISTRY.build(cfg.model.name, cfg.model)
        ```

        Use Hydra instantiate only where it improves clarity.
    """,
    )


def create_code_walkthroughs() -> None:
    """Create detailed code walkthroughs that explain which lines cause each Hydra behavior."""
    section('Detailed lesson walkthroughs — why each code/config line matters')

    lessons = [
        '00_basics',
        '01_config_groups',
        '02_overrides',
        '03_interpolation',
        '04_structured_configs',
        '05_instantiate',
        '06_multirun',
        '07_plugins',
        '08_real_world_ml',
    ]
    for lesson in lessons:
        append_file(
            f'{lesson}/LESSON.md',
            """
            ---

            ## Read next

            For a slower explanation of exactly which code/config lines create
            the behavior in this lesson, read:

            ```bash
            cat CODE_WALKTHROUGH.md
            ```
        """,
        )

    append_file(
        'README.md',
        """
        ## Added explanatory walkthroughs

        Each lesson folder now includes:

        ```text
        CODE_WALKTHROUGH.md
        ```

        This file explains which exact Python/YAML lines cause the Hydra
        behavior in that lesson. Also read:

        ```text
        PHD_RESEARCH_GUIDE.md
        ```

        for a practical subset of Hydra features that is enough for most
        research ML code.
    """,
    )

    write(
        '00_basics/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 0 Code Walkthrough — Basics

        ## What this lesson is teaching

        Hydra is not doing machine learning yet. It is doing only one thing:
        loading `conf/config.yaml`, turning it into `cfg`, and passing `cfg`
        into your Python function.

        ## Files involved

        ```text
        00_basics/
        ├── app.py
        └── conf/config.yaml
        ```

        ## The config file is the source of values

        ```yaml
        db:
          host: localhost
          port: 5432

        training:
          lr: 0.001
          epochs: 10
        ```

        This creates a tree:

        ```text
        cfg
        ├── db
        │   ├── host
        │   └── port
        └── training
            ├── lr
            └── epochs
        ```

        ## The decorator connects Python to YAML

        ```python
        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig) -> None:
        ```

        The important parts are:

        | Code | Meaning |
        |---|---|
        | `config_path="conf"` | Look inside the `conf/` folder |
        | `config_name="config"` | Load `conf/config.yaml` |
        | `cfg: DictConfig` | Receive the loaded config as an OmegaConf object |

        ## This line prints the full config

        ```python
        print(OmegaConf.to_yaml(cfg))
        ```

        This is your first debugging tool. Before training, always verify what
        Hydra actually composed.

        ## These lines prove dot access

        ```python
        print(cfg.db.host)
        print(cfg.training.lr)
        ```

        They correspond directly to:

        ```yaml
        db:
          host: localhost
        training:
          lr: 0.001
        ```

        ## CLI overrides

        ```bash
        python app.py training.lr=0.01
        ```

        This does not edit the YAML file. It creates a temporary final config
        for this run where:

        ```yaml
        training:
          lr: 0.01
        ```

        ## `--cfg job`

        ```bash
        python app.py --cfg job
        ```

        `job` means the application config that your code receives. It is not a
        key you created. It is a Hydra debug mode.
    """,
    )

    write(
        '01_config_groups/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 1 Code Walkthrough — Config Groups

        ## What this lesson is teaching

        Config groups let you swap whole blocks of configuration from the CLI.
        This is the core Hydra pattern for ML experiments.

        ## Files involved

        ```text
        01_config_groups/conf/
        ├── config.yaml
        ├── db/postgres.yaml
        ├── db/sqlite.yaml
        ├── model/small.yaml
        ├── model/large.yaml
        └── optimizer/adam.yaml
        ```

        ## The `defaults` list chooses files

        ```yaml
        defaults:
          - db: postgres
          - model: small
          - optimizer: adam
          - _self_
        ```

        This means:

        | Defaults entry | File loaded |
        |---|---|
        | `db: postgres` | `conf/db/postgres.yaml` |
        | `model: small` | `conf/model/small.yaml` |
        | `optimizer: adam` | `conf/optimizer/adam.yaml` |

        The identifiers are not inside the files. They come from:

        ```text
        folder name = group name
        file name   = option name
        ```

        So `model=large` means “replace `conf/model/small.yaml` with
        `conf/model/large.yaml`.”

        ## What `@package _global_` is doing here

        Example group config:

        ```yaml
        # @package _global_
        model:
          name: transformer_small
          hidden_size: 128
        ```

        `_global_` says: merge this file exactly at the root of the final tree.
        Because the file already contains `model:`, Hydra should not wrap it
        again under `model:`.

        If the file had no `@package _global_`, this exact YAML would become:

        ```yaml
        model:
          model:
            name: transformer_small
        ```

        That is why this tutorial uses `_global_` with files that already write
        the top-level key themselves.

        ## What `_self_` controls

        ```yaml
        - _self_
        ```

        means “merge this root `config.yaml` at this position.” Because it is
        last, root keys like `experiment_name`, `seed`, and `debug` merge after
        the group configs.

        `_self_` controls merge precedence, not tree location.

        ## Good organization rule

        Avoid overlapping ownership:

        ```text
        db/          owns db.*
        model/       owns model.*
        optimizer/   owns optimizer.*
        config.yaml  mainly composes groups and stores global metadata
        ```

        If many files define the same key, the final result may be correct, but
        the config becomes harder to reason about.
    """,
    )

    write(
        '02_overrides/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 2 Code Walkthrough — CLI Overrides

        ## What this lesson is teaching

        CLI overrides are temporary edits to the composed config tree. They do
        not modify YAML files on disk.

        ## The starting tree

        ```yaml
        app:
          name: myapp
          debug: false
          tags: [production, v1]

        server:
          host: 0.0.0.0
          port: 8080
        ```

        ## Existing-key update

        ```bash
        python demo.py app.name=prod_app
        ```

        This works because `app.name` already exists.

        ## `+` vs `++`

        | Operator | If key exists | If key does not exist | Meaning |
        |---|---:|---:|---|
        | `key=value` | Works | Error | Update existing only |
        | `+key=value` | Error | Works | Add new only |
        | `++key=value` | Works | Works | Upsert: update or add |

        Try these:

        ```bash
        python demo.py +app.version=2.0       # works: version did not exist
        python demo.py +app.name=other        # fails: name already exists
        python demo.py ++app.name=other       # works: updates existing
        python demo.py ++app.version=2.0      # works: creates missing
        ```

        ## Inline dict syntax uses colons, not equals signs

        Correct:

        ```bash
        python demo.py '+server={host:127.0.0.1,port:9090,ssl:true}'
        ```

        Wrong:

        ```bash
        python demo.py '+server={host:127.0.0.1,port:9090,ssl=true}'
        ```

        Inside Hydra's inline dict, use:

        ```yaml
        key:value
        ```

        not:

        ```yaml
        key=value
        ```

        ## Shell quoting equivalences

        These two pass the same override to Hydra:

        ```bash
        python demo.py 'app.tags=[staging,v2,hotfix]'
        python demo.py app.tags="[staging,v2,hotfix]"
        ```

        These are equivalent for the colon string:

        ```bash
        python demo.py 'app.name="hello:world"'
        python demo.py app.name="hello:world"
        python demo.py 'app.name=hello:world'
        ```

        These are equivalent for the inline dict:

        ```bash
        python demo.py '+server={host:127.0.0.1,port:9090,ssl:true}'
        python demo.py +server="{host:127.0.0.1,port:9090,ssl:true}"
        ```

        ## Why the Python code uses `open_dict`

        ```python
        with open_dict(cfg):
            cfg.app.runtime_id = "abc123"
        ```

        Hydra/OmegaConf usually protects the config from accidental unknown keys.
        `open_dict` temporarily unlocks it so the code can add `runtime_id`.
        This is Python-side mutation, separate from CLI overrides.
    """,
    )

    write(
        '03_interpolation/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 3 Code Walkthrough — Interpolation and Resolvers

        ## What this lesson is teaching

        Interpolation stores formulas in YAML. The formula is resolved when the
        value is accessed.

        ## Basic interpolation

        ```yaml
        project: hydra_demo
        version: "1.0"
        env: development
        run_name: "${project}_${version}_${env}"
        ```

        `run_name` is not permanently stored as `hydra_demo_1.0_development`.
        It is stored as a reference expression. When you access `cfg.run_name`,
        OmegaConf resolves the current values of `project`, `version`, and `env`.

        ## Path interpolation

        ```yaml
        base_dir: /workspace
        output_dir: ${base_dir}/outputs/${project}
        checkpoint_dir: ${output_dir}/checkpoints
        model:
          save_path: "${checkpoint_dir}/${model.name}.pt"
        ```

        Command:

        ```bash
        python app.py base_dir=/data/experiments
        ```

        changes `base_dir`; all dependent paths update because they are resolved
        lazily.

        ## `oc.env` means OS environment variable

        ```yaml
        api_key: ${oc.env:API_KEY,not_set}
        ```

        `oc.env` is a built-in OmegaConf resolver. It tells OmegaConf:

        ```python
        os.environ.get("API_KEY", "not_set")
        ```

        It is different from:

        ```yaml
        api_key: ${API_KEY}
        ```

        because `${API_KEY}` would look for a config key named `API_KEY`.

        ## Custom resolver registration

        ```python
        OmegaConf.register_new_resolver(
            "choose_log_level",
            lambda env: "DEBUG" if env == "development" else "INFO"
        )
        ```

        This registers a new YAML function:

        ```yaml
        log_level: ${choose_log_level:${env}}
        ```

        If `env=development`, this becomes:

        ```python
        choose_log_level("development") -> "DEBUG"
        ```

        If `env=production`, it becomes:

        ```python
        choose_log_level("production") -> "INFO"
        ```

        It is not selecting another config file. It is just calling a Python
        function during interpolation resolution.

        ## What `~project` does

        ```bash
        python app.py '~project'
        ```

        deletes the key `project` from the config. Then this line breaks when
        accessed:

        ```yaml
        run_name: "${project}_${version}_${env}"
        ```

        because `${project}` no longer exists.

        ## `oc.select` means safe lookup with fallback

        ```yaml
        safe_value: ${oc.select:maybe.missing.key,fallback_value}
        ```

        This means: try to read `maybe.missing.key`; if it does not exist,
        return `fallback_value` instead of raising an interpolation error.

        It is useful for optional config blocks.
    """,
    )

    write(
        '04_structured_configs/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 4 Code Walkthrough — Structured Configs

        ## What this lesson is teaching

        Structured configs are a schema system. They are optional, and for fast
        PhD research they may be overkill. The main value is catching bad config
        before a long run starts.

        ## The schema file

        ```python
        @dataclass
        class TrainingConfig:
            epochs: int = 10
            batch_size: int = 32
            mixed_precision: bool = False
        ```

        This does three things:

        | Part | Meaning |
        |---|---|
        | `epochs: int` | type rule: epochs must be int |
        | `= 10` | schema default value |
        | dataclass field name | allowed key name |

        ## `AppConfig` is the root schema

        ```python
        @dataclass
        class AppConfig:
            db: DBConfig = field(default_factory=DBConfig)
            model: ModelConfig = field(default_factory=ModelConfig)
            optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
            training: TrainingConfig = field(default_factory=TrainingConfig)
            experiment_name: str = MISSING
        ```

        `AppConfig` describes the whole final config tree:

        ```text
        cfg
        ├── db: DBConfig
        ├── model: ModelConfig
        ├── optimizer: OptimizerConfig
        ├── training: TrainingConfig
        └── experiment_name: str
        ```

        ## What `MISSING` does

        ```python
        experiment_name: str = MISSING
        ```

        means: there is no valid default. The user must provide this value in
        YAML or from CLI.

        In the composed config, this is internally represented like:

        ```yaml
        experiment_name: ???
        ```

        This lesson forces missing checks at startup with:

        ```python
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        ```

        ## Registering does not activate the schema

        ```python
        cs = ConfigStore.instance()
        cs.store(name="app_schema", node=AppConfig)
        ```

        This only puts the schema in Hydra's registry. It is like saying:

        ```text
        Hydra, here is a blueprint named app_schema.
        ```

        It does not mean the blueprint is used.

        ## This line activates the schema

        ```yaml
        defaults:
          - app_schema
          - _self_
        ```

        This means:

        ```text
        1. Start with AppConfig defaults and type rules.
        2. Merge this YAML file on top.
        ```

        So schema defaults are initial values, and YAML values override them.

        Example:

        ```python
        epochs: int = 10
        ```

        plus YAML:

        ```yaml
        training:
          epochs: 20
        ```

        final result:

        ```yaml
        training:
          epochs: 20
        ```

        But the type rule still remains: `epochs` must be an int.

        ## Why your earlier version did not validate

        If `config.yaml` does not include:

        ```yaml
        - app_schema
        ```

        then Hydra just loads plain YAML. That is why this could slip through:

        ```bash
        python app.py training.epochs=not_an_int
        ```

        and become a string.

        ## Research recommendation

        Use structured configs only for stable parts:

        ```text
        training
        optimizer
        checkpointing
        logging
        reproducibility
        ```

        Keep rapidly changing model/dataset experiments as plain YAML if that is
        faster for exploration.
    """,
    )

    write(
        '05_instantiate/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 5 Code Walkthrough — instantiate() vs Registry

        ## What this lesson is teaching

        Hydra can create Python objects from YAML using `_target_`. This is
        powerful, but for research code a registry is often easier to debug.

        ## `_target_` means import path

        ```yaml
        optimizer:
          _target_: components.FakeAdam
          lr: 0.001
          betas: [0.9, 0.999]
        ```

        When Python runs:

        ```python
        optimizer = instantiate(cfg.optimizer)
        ```

        Hydra does roughly:

        ```python
        from components import FakeAdam
        optimizer = FakeAdam(lr=0.001, betas=[0.9, 0.999])
        ```

        ## Constructor arguments come from YAML keys

        Class:

        ```python
        class FakeAdam:
            def __init__(self, lr, betas, weight_decay=0.0):
        ```

        YAML:

        ```yaml
        lr: 0.001
        betas: [0.9, 0.999]
        weight_decay: 0.0
        ```

        These names must match the constructor arguments.

        ## `_partial_=True`

        ```python
        opt_partial = instantiate(cfg.optimizer, _partial_=True)
        ```

        This does not create the object yet. It returns a factory function.
        This is useful when part of the constructor is known only at runtime.

        For real PyTorch:

        ```python
        opt_fn = instantiate(cfg.optimizer, _partial_=True)
        optimizer = opt_fn(params=model.parameters())
        ```

        ## `_convert_="all"`

        Hydra values are often `DictConfig` or `ListConfig`, not plain dict/list.
        Some libraries expect plain Python containers. Then use:

        ```python
        instantiate(cfg.object, _convert_="all")
        ```

        ## Why Registry may be better for your ML template

        Registry style:

        ```yaml
        model:
          name: resnet50
          hidden_size: 256
        ```

        ```python
        model = MODEL_REGISTRY.build(cfg.model.name, cfg.model)
        ```

        Advantages:

        ```text
        easier stack traces
        no magic dynamic imports in YAML
        simpler refactoring
        easier to search code
        better for custom research models
        ```

        Practical recommendation:

        ```text
        Models/datasets/losses/metrics -> Registry
        Optimizers/schedulers          -> Registry or instantiate, your choice
        ```
    """,
    )

    write(
        '06_multirun/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 6 Code Walkthrough — Multirun and Sweeps

        ## What this lesson is teaching

        `-m` means multirun/sweep mode. It is not needed for normal one-run
        training.

        ## Normal run: no `-m`

        ```bash
        python train.py model=small optimizer=adam
        ```

        This creates one composed config and runs once.

        ## Sweep run: use `-m`

        ```bash
        python train.py -m optimizer.lr=0.1,0.01,0.001
        ```

        The comma-separated value is a sweep expression. Hydra expands it into:

        ```text
        job 0: optimizer.lr=0.1
        job 1: optimizer.lr=0.01
        job 2: optimizer.lr=0.001
        ```

        ## Cartesian product

        ```bash
        python train.py -m model=small,large optimizer=adam,sgd
        ```

        means:

        ```text
        2 models x 2 optimizers = 4 jobs
        ```

        ## `glob(*)`

        ```bash
        python train.py -m 'model=glob(*)'
        ```

        means: look inside `conf/model/` and sweep over every available config
        option. If the folder contains only `small.yaml` and `large.yaml`, Hydra
        launches 2 jobs. That is still multirun.

        ## Where the Hydra log lines come from

        These lines:

        ```text
        [HYDRA] Launching 3 jobs locally
        [HYDRA] #0 : optimizer.lr=0.1
        ```

        are Hydra's default logging, not your code.

        Your code starts here:

        ```python
        print(f"\n[Run {job_num}] {cfg.model.name} + {cfg.optimizer.name}")
        ```

        ## How the script knows the run number

        ```python
        hcfg = HydraConfig.get()
        job_num = hcfg.job.num
        overrides = hcfg.overrides.task
        ```

        This reads Hydra's runtime metadata for the current job.

        ## Default launcher

        By default Hydra uses the basic launcher, so jobs run locally and usually
        sequentially. You can later switch to joblib/SLURM/Ray launchers without
        rewriting your training loop.
    """,
    )

    write(
        '07_plugins/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 7 Code Walkthrough — Advanced Patterns

        ## What this lesson is teaching

        This lesson shows Hydra features that are useful, but not required for
        every research project.

        ## Overriding Hydra's own config

        ```yaml
        defaults:
          - _self_
          - override hydra/output: custom
        ```

        `hydra/output` is not your application config. It is part of Hydra's
        internal config tree. This line replaces Hydra's default output-dir
        policy with `conf/hydra/output/custom.yaml`.

        ## Why paths can be confusing

        Hydra changes the current working directory to the run directory. So:

        ```python
        os.getcwd()
        ```

        may no longer be your project folder.

        Use:

        ```python
        from hydra.utils import get_original_cwd, to_absolute_path
        ```

        `to_absolute_path("data/train.csv")` resolves relative to the original
        directory where you launched the program.

        ## Compose API

        ```python
        with initialize(config_path="conf", version_base=None):
            cfg = compose(config_name="config")
        ```

        This lets you use Hydra in scripts/tests without decorating a main
        function. It is useful for unit tests around config composition.

        ## Research recommendation

        Learn these, but do not overuse them early. The most valuable advanced
        feature for research is usually path handling with `to_absolute_path`.
    """,
    )

    write(
        '08_real_world_ml/CODE_WALKTHROUGH.md',
        r"""
        # Lesson 8 Code Walkthrough — Real-World ML Layout

        ## What this lesson is teaching

        This folder combines the patterns into a realistic ML experiment layout.

        ## The root config composes components

        ```yaml
        defaults:
          - dataset: mnist
          - model: cnn
          - optimizer: adamw
          - scheduler: cosine
          - callbacks: default
          - logger: console
          - training
          - _self_
        ```

        This is the Hydra equivalent of selecting experiment components.

        ## Required experiment name

        ```yaml
        experiment_name: ???
        ```

        `???` means missing mandatory value in plain OmegaConf/YAML style. The
        run should provide it:

        ```bash
        python train.py experiment_name=baseline
        ```

        ## Cross-config interpolation

        In model config:

        ```yaml
        num_classes: ${dataset.num_classes}
        ```

        If you switch:

        ```bash
        python train.py experiment_name=x dataset=cifar10 model=resnet
        ```

        then `model.num_classes` follows the selected dataset.

        ## Hydra output dir customization

        ```yaml
        hydra:
          run:
            dir: ${output_dir}
        ```

        This tells Hydra where to create the run directory.

        ## Note on `_target_`

        This lesson uses `_target_` to show the Hydra-native style. For your own
        research template, you can replace this with a Registry pattern:

        ```python
        dataset = DATASET_REGISTRY.build(cfg.dataset.name, cfg.dataset)
        model = MODEL_REGISTRY.build(cfg.model.name, cfg.model)
        ```

        The config group idea remains exactly the same either way.
    """,
    )

    write(
        'PHD_RESEARCH_GUIDE.md',
        r"""
        # Practical Hydra Subset for PhD Research Code

        You do not need every Hydra feature.

        ## Use these early

        ```text
        config groups
        defaults list
        CLI overrides
        interpolation
        --cfg job
        --info defaults
        multirun sweeps with -m
        ```

        ## Use with caution

        ```text
        structured configs
        custom resolvers
        compose API
        Hydra internal overrides
        ```

        ## Probably prefer Registry over `_target_`

        For research models/datasets:

        ```python
        model = MODEL_REGISTRY.build(cfg.model.name, cfg.model)
        dataset = DATASET_REGISTRY.build(cfg.dataset.name, cfg.dataset)
        ```

        This is easier to debug than dynamic imports in YAML.

        ## Suggested architecture

        ```text
        conf/
        ├── config.yaml
        ├── model/
        ├── dataset/
        ├── optimizer/
        ├── scheduler/
        ├── trainer/
        └── logger/
        src/
        ├── registries.py
        ├── models/
        ├── datasets/
        ├── training/
        └── utils/
        ```

        ## Minimal rule

        Keep each config group responsible for one section:

        ```text
        model/ owns model.*
        dataset/ owns dataset.*
        optimizer/ owns optimizer.*
        ```

        Avoid defining the same key in many places unless the override is very
        intentional.
    """,
    )

    note('Created: CODE_WALKTHROUGH.md files for every lesson')


def print_tutorial() -> None:
    """Print the recommended tutorial study sequence."""
    banner('HYDRA TUTORIAL — STUDY GUIDE (2-3 hours)', '═')

    print("""
  Work through the lessons in order. Each builds on the last.
  Every folder has a LESSON.md — read it and run the commands.

  ╔══════════════════════════════════════════════════════════════╗
  ║  LESSON ORDER                                               ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  0 │ 00_basics           ~15 min  Core mechanics           ║
  ║  1 │ 01_config_groups    ~20 min  Swapping config blocks   ║
  ║  2 │ 02_overrides        ~15 min  All override operators   ║
  ║  3 │ 03_interpolation    ~20 min  Variable refs & resolvers║
  ║  4 │ 04_structured_cfg   ~20 min  Typed schemas            ║
  ║  5 │ 05_instantiate      ~20 min  Build objects from YAML  ║
  ║  6 │ 06_multirun         ~15 min  Sweep experiments        ║
  ║  7 │ 07_plugins          ~15 min  Advanced & compose API   ║
  ║  8 │ 08_real_world_ml    ~20 min  Full ML project          ║
  ╚══════════════════════════════════════════════════════════════╝
""")

    banner('LESSON 0 — BASICS', '─')
    print("""
  The anatomy of a Hydra app:

    @hydra.main(
        version_base=None,      ← suppress future-version warnings
        config_path="conf",     ← folder containing YAML files
        config_name="config"    ← filename without .yaml
    )
    def main(cfg: DictConfig):
        print(cfg.db.host)      ← dot notation access

  Running:
    cd hydra_tutorial/00_basics
    python app.py                              # default
    python app.py db.host=prod db.port=5433   # override
    python app.py --cfg job                    # print config, don't run
""")

    banner('LESSON 1 — CONFIG GROUPS', '─')
    print("""
  Config groups = swappable YAML blocks.

  conf/
  ├── config.yaml          ← root: lists defaults
  ├── db/
  │   ├── postgres.yaml
  │   └── sqlite.yaml
  └── model/
      ├── small.yaml
      └── large.yaml

  config.yaml:
    defaults:
      - db: postgres     ← load db/postgres.yaml
      - model: small     ← load model/small.yaml
      - _self_

  CLI:
    python train.py db=sqlite             # swap db group
    python train.py model=large           # swap model group
    python train.py db=sqlite model=large # both at once
    python train.py --info defaults       # see what was loaded
""")

    banner('LESSON 2 — OVERRIDES', '─')
    print("""
  Override operators:
    key=value       update existing key only
    +key=value      add new key only; error if it exists
    ++key=value     upsert: update or add
    ~key            delete key
    key=null        set to None
    'key=[a,b,c]'   set list (quote for shell)

  Key table:
    key=value       exists -> update, missing -> error
    +key=value      exists -> error,  missing -> add
    ++key=value     exists -> update, missing -> add

  Quoting equivalence examples:
    python demo.py 'app.tags=[staging,v2,hotfix]'
    python demo.py app.tags="[staging,v2,hotfix]"
    python demo.py '+server={host:127.0.0.1,port:9090,ssl:true}'
    python demo.py +server="{host:127.0.0.1,port:9090,ssl:true}"
""")

    banner('LESSON 3 — INTERPOLATION', '─')
    print("""
  Reference other config values with ${key.path}:

    base_dir: /workspace
    data_dir: ${base_dir}/data     ← resolves lazily

  Override base_dir → data_dir updates automatically!

  Built-in resolvers:
    ${oc.env:HOME}                 ← os.environ['HOME']
    ${oc.env:SECRET,default}       ← with fallback

  Custom resolvers (register before @hydra.main):
    OmegaConf.register_new_resolver("upper", str.upper)
    # YAML: ${upper:${project}}
""")

    banner('LESSON 4 — STRUCTURED CONFIGS', '─')
    print("""
  Use dataclasses for typed schemas:

    from omegaconf import MISSING
    @dataclass
    class DBConfig:
        host: str = "localhost"
        port: int = 5432
        name: str = MISSING    ← required, no default

  Structured configs are optional and can be overkill for research.

  They only work if the schema is ACTIVE:
    cs = ConfigStore.instance()
    cs.store(name="app_schema", node=AppConfig)

    # conf/config.yaml
    defaults:
      - app_schema
      - _self_

  AppConfig provides default values + type rules.
  YAML values override those defaults but must match the schema.
""")

    banner('LESSON 5 — INSTANTIATE', '─')
    print("""
  Build Python objects directly from YAML:

    # config.yaml
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.001
      weight_decay: 1e-4

    # Python
    opt = instantiate(cfg.optimizer, params=model.parameters())

  _partial_=True → returns functools.partial (call later):
    opt_fn = instantiate(cfg.optimizer, _partial_=True)
    opt = opt_fn(params=model.parameters())

  Swap the entire optimizer class from CLI:
    python train.py optimizer=sgd
""")

    banner('LESSON 6 — MULTIRUN', '─')
    print("""
  Normal single runs do NOT need -m:

    python train.py model=large optimizer.lr=0.0003

  Use -m only for sweep/multirun expressions:

    python train.py -m optimizer.lr=0.1,0.01,0.001
    # → 3 runs with lr=0.1, lr=0.01, lr=0.001

    python train.py -m model=small,large optimizer=adam,sgd
    # → 4 runs (2x2 grid)

    python train.py -m 'optimizer.lr=range(0.001,0.1,0.01)'
    # → range sweep

  Each run gets:
    multirun/DATE/TIME/0/   ← run 0
    multirun/DATE/TIME/1/   ← run 1
    ...each with its own .hydra/config.yaml

  Parallel runs (after pip install hydra-joblib-launcher):
    python train.py -m hydra/launcher=joblib hydra.launcher.n_jobs=4 ...
""")

    banner('LESSON 7 — ADVANCED', '─')
    print("""
  @package directive — control where keys land:
    # @package _global_       root of tree
    # @package model.encoder  under model.encoder

  compose API — use Hydra in tests/scripts (no decorator):
    from hydra import compose, initialize
    with initialize(config_path="conf", version_base=None):
        cfg = compose("config", overrides=["db=sqlite"])

  Path helpers (Hydra changes your CWD!):
    from hydra.utils import get_original_cwd, to_absolute_path
    path = to_absolute_path(cfg.data_dir)  # always correct

  Disable CWD change:
    python app.py 'hydra.output_subdir=null' hydra.run.dir=.
""")

    banner('LESSON 8 — REAL WORLD ML', '─')
    print("""
  Everything combined in a single project.
  Key patterns:
    • All components have _target_ → instantiate them
    • Cross-config interpolation (model gets num_classes from dataset)
    • experiment_name: ??? forces you to always name your run
    • Multirun sweep across datasets x models x optimizers

  Try:
    cd hydra_tutorial/08_real_world_ml
    python train.py experiment_name=my_first_run
    python train.py experiment_name=cifar dataset=cifar10 model=resnet
    python train.py -m experiment_name=sweep \\
        dataset=mnist,cifar10 model=cnn,resnet optimizer=adamw,sgd
""")

    banner('DONE! Files created. Start with:', '═')
    print("""
    cd hydra_tutorial/00_basics
    python app.py
    python app.py --cfg job
    python app.py db.host=myserver training.lr=0.01

  Then open LESSON.md in each folder and run the commands!
  Happy learning 🚀
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Install dependencies and generate the complete tutorial repository."""
    banner('HYDRA TUTORIAL REPO SETUP')
    print(f'\n  Creating repo at: {ROOT.resolve()}\n')

    install_hydra()

    ROOT.mkdir(exist_ok=True, parents=True)

    create_00_basics()
    create_01_config_groups()
    create_02_overrides()
    create_03_interpolation()
    create_04_structured_configs()
    create_05_instantiate()
    create_06_multirun()
    create_07_plugins()
    create_08_real_world()
    create_readme()
    create_code_walkthroughs()

    banner('ALL FILES CREATED ✅')
    print(f'\n  Repo location: {ROOT.resolve()}')

    # Count files
    total = sum(len(files) for _, _, files in os.walk(ROOT))
    print(f'  Total files  : {total}')

    print_tutorial()


if __name__ == '__main__':
    main()
