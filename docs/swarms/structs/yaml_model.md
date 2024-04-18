# YamlModel: A Pydantic Model for YAML Data

### Introduction

The `YamlModel` class, derived from `BaseModel` in Pydantic, offers a convenient way to work with YAML data in your Python applications. It provides methods for serialization (converting to YAML), deserialization (creating an instance from YAML), and schema generation. This documentation will delve into the functionalities of `YamlModel` and guide you through its usage with illustrative examples.

### Purpose and Functionality

The primary purpose of `YamlModel` is to streamline the interaction between your Python code and YAML data. It accomplishes this by:

* **Serialization:** Transforming a `YamlModel` instance into a YAML string representation using the `to_yaml()` method.
* **Deserialization:** Constructing a `YamlModel` instance from a provided YAML string using the `from_yaml()` class method.
* **JSON to YAML Conversion:** Facilitating the conversion of JSON data to YAML format through the `json_to_yaml()` static method.
* **Saving to YAML File:** Enabling the storage of `YamlModel` instances as YAML files using the `save_to_yaml()` method.
* (Future Implementation) **Schema Generation:** The `create_yaml_schema()` class method (not yet implemented but included for future reference) will generate a YAML schema that reflects the structure of the `YamlModel` class and its fields.

### Class Definition and Arguments

The `YamlModel` class inherits from Pydantic's `BaseModel` class. You can define your custom YAML models by creating subclasses of `YamlModel` and specifying your data fields within the class definition. Here's the breakdown of the `YamlModel` class and its methods:

```python
class YamlModel(BaseModel):
    """
    A Pydantic model class for working with YAML data.
    """

    def to_yaml(self):
        """
        Serialize the Pydantic model instance to a YAML string.
        """
        return yaml.safe_dump(self.dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str):
        """
        Create an instance of the class from a YAML string.

        Args:
            yaml_str (str): The YAML string to parse.

        Returns:
            cls: An instance of the class with attributes populated from the YAML data.
                 Returns None if there was an error loading the YAML data.
        """
        # ...

    @staticmethod
    def json_to_yaml(json_str: str):
        """
        Convert a JSON string to a YAML string.
        """
        # ...

    def save_to_yaml(self, filename: str):
        """
        Save the Pydantic model instance as a YAML file.
        """
        # ...

    # TODO: Implement a method to create a YAML schema from the model fields
    # @classmethod
    # def create_yaml_schema(cls):
    #     # ...
    """
```

**Arguments:**

* `self` (implicit): Refers to the current instance of the `YamlModel` class.
* `yaml_str` (str): The YAML string used for deserialization in the `from_yaml()` method.
* `json_str` (str): The JSON string used for conversion to YAML in the `json_to_yaml()` method.
* `filename` (str): The filename (including path) for saving the YAML model instance in the `save_to_yaml()` method.

### Detailed Method Descriptions

**1. to_yaml()**

This method transforms an instance of the `YamlModel` class into a YAML string representation. It utilizes the `yaml.safe_dump()` function from the `PyYAML` library to ensure secure YAML data generation. The `sort_keys=False` argument guarantees that the order of keys in the resulting YAML string remains consistent with the order of fields defined in your `YamlModel` subclass.

**Example:**

```python
class User(YamlModel):
    name: str
    age: int
    is_active: bool

user = User(name="Bob", age=30, is_active=True)
yaml_string = user.to_yaml()
print(yaml_string)
```

This code will output a YAML string representation of the `user` object, resembling:

```yaml
name: Bob
age: 30
is_active: true
```

### Detailed Method Descriptions 

**2. from_yaml(cls, yaml_str)** (Class Method)

The `from_yaml()` class method is responsible for constructing a `YamlModel` instance from a provided YAML string. 

* **Arguments:**
    * `cls` (class): The class representing the desired YAML model (the subclass of `YamlModel` that matches the structure of the YAML data).
    * `yaml_str` (str): The YAML string containing the data to be parsed and used for creating the model instance.

* **Returns:**
    * `cls` (instance): An instance of the specified class (`cls`) populated with the data extracted from the YAML string. If an error occurs during parsing, it returns `None`.

* **Error Handling:**

The `from_yaml()` method employs `yaml.safe_load()` for secure YAML parsing. It incorporates a `try-except` block to handle potential `ValueError` exceptions that might arise during the parsing process. If an error is encountered, it logs the error message and returns `None`.

**Example:**

```python
class User(YamlModel):
    name: str
    age: int
    is_active: bool

yaml_string = """
name: Alice
age: 25
is_active: false
"""

user = User.from_yaml(yaml_string)
print(user.name)  # Output: Alice
```

**3. json_to_yaml(json_str)** (Static Method)

This static method in the `YamlModel` class serves the purpose of converting a JSON string into a YAML string representation.

* **Arguments:**
    * `json_str` (str): The JSON string that needs to be converted to YAML format.

* **Returns:**
    * `str`: The converted YAML string representation of the provided JSON data.

* **Functionality:**

The `json_to_yaml()` method leverages the `json.loads()` function to parse the JSON string into a Python dictionary. Subsequently, it utilizes `yaml.dump()` to generate the corresponding YAML string representation from the parsed dictionary.

**Example:**

```python
json_string = '{"name": "Charlie", "age": 42, "is_active": true}'
yaml_string = YamlModel.json_to_yaml(json_string)
print(yaml_string)
```

This code snippet will convert the JSON data to a YAML string, likely resembling:

```yaml
name: Charlie
age: 42
is_active: true
```

**4. save_to_yaml(self, filename)**

The `save_to_yaml()` method facilitates the storage of a `YamlModel` instance as a YAML file.

* **Arguments:**
    * `self` (implicit): Refers to the current instance of the `YamlModel` class that you intend to save.
    * `filename` (str): The desired filename (including path) for the YAML file.

* **Functionality:**

The `save_to_yaml()` method employs the previously explained `to_yaml()` method to generate a YAML string representation of the `self` instance. It then opens the specified file in write mode (`"w"`) and writes the YAML string content to the file.

**Example:**

```python
class Employee(YamlModel):
    name: str
    department: str
    salary: float

employee = Employee(name="David", department="Engineering", salary=95000.00)
employee.save_to_yaml("employee.yaml")
```

This code will create a YAML file named "employee.yaml" containing the serialized representation of the `employee` object.


### More Usage Examples ++ 

```python
class User(YamlModel):
    name: str
    age: int
    is_active: bool

# Create an instance of the User model
user = User(name="Alice", age=30, is_active=True)

# Serialize the User instance to YAML and print it
yaml_string = user.to_yaml()
print(yaml_string)
```

This code snippet demonstrates the creation of a `User` instance and its subsequent serialization to a YAML string using the `to_yaml()` method. The printed output will likely resemble:

```yaml
name: Alice
age: 30
is_active: true
```

### Converting JSON to YAML

```python
# Convert JSON string to YAML and print
json_string = '{"name": "Bob", "age": 25, "is_active": false}'
yaml_string = YamlModel.json_to_yaml(json_string)
print(yaml_string)
```

This example showcases the conversion of a JSON string containing user data into a YAML string representation using the `json_to_yaml()` static method. The resulting YAML string might look like:

```yaml
name: Bob
age: 25
is_active: false
```

### Saving User Instance to YAML File

```python
# Save the User instance to a YAML file
user.save_to_yaml("user.yaml")
```

This code demonstrates the utilization of the `save_to_yaml()` method to store the `user` instance as a YAML file named "user.yaml". The contents of the file will mirror the serialized YAML string representation of the user object.

## Additional Considerations

* Ensure you have the `PyYAML` library installed (`pip install pyyaml`) to leverage the YAML parsing and serialization functionalities within `YamlModel`.
* Remember that the `create_yaml_schema()` method is not yet implemented but serves as a placeholder for future enhancements.
* For complex data structures within your YAML models, consider leveraging Pydantic's data validation and nested model capabilities for robust data management.

## Conclusion

The `YamlModel` class in Pydantic offers a streamlined approach to working with YAML data in your Python projects. By employing the provided methods (`to_yaml()`, `from_yaml()`, `json_to_yaml()`, and `save_to_yaml()`), you can efficiently convert between Python objects and YAML representations, facilitating data persistence and exchange. This comprehensive documentation empowers you to effectively utilize `YamlModel` for your YAML data processing requirements.