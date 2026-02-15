# Base Structure Tutorial

    End-to-end usage for `base_structure`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms.structs.base_structure import BaseStructure

base = BaseStructure(name="demo-structure")
base.save_to_file({"status": "ok"}, "./artifacts/demo.json")
loaded = base.load_from_file("./artifacts/demo.json")
print(loaded)

base.log_event("Saved demo artifact")
base.log_error("Example error message")
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `base_structure`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/base_structure.md`](../structs/base_structure.md)
