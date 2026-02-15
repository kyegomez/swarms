# BaseStructure Tutorial

    End-to-end tutorial for `swarms.structs.base_structure`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms.structs.base_structure import BaseStructure

base = BaseStructure(name="demo", save_metadata_on=False)
base.save_to_file({"status": "ok"}, "./demo.json")
print(base.load_from_file("./demo.json"))
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `base_structure`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/base_structure.md`](../structs/base_structure.md)
