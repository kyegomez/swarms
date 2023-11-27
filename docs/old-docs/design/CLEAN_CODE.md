Code is clean if it can be understood easily – by everyone on the team. Clean code can be read and enhanced by a developer other than its original author. With understandability comes readability, changeability, extensibility and maintainability.
_____________________________________

## General rules
1. Follow standard conventions.
2. Keep it simple stupid. Simpler is always better. Reduce complexity as much as possible.
3. Boy scout rule. Leave the campground cleaner than you found it.
4. Always find root cause. Always look for the root cause of a problem.

## Design rules
1. Keep configurable data at high levels.
2. Prefer polymorphism to if/else or switch/case.
3. Separate multi-threading code.
4. Prevent over-configurability.
5. Use dependency injection.
6. Follow Law of Demeter. A class should know only its direct dependencies.

## Understandability tips
1. Be consistent. If you do something a certain way, do all similar things in the same way.
2. Use explanatory variables.
3. Encapsulate boundary conditions. Boundary conditions are hard to keep track of. Put the processing for them in one place.
4. Prefer dedicated value objects to primitive type.
5. Avoid logical dependency. Don't write methods which works correctly depending on something else in the same class.
6. Avoid negative conditionals.

## Names rules
1. Choose descriptive and unambiguous names.
2. Make meaningful distinction.
3. Use pronounceable names.
4. Use searchable names.
5. Replace magic numbers with named constants.
6. Avoid encodings. Don't append prefixes or type information.
7. The Name of a variable, Function, or Class should answer  why it exists, what it does , and how it can used. Comments are a burden
8. Clarity is King
9. ClassNames should not be a verb
10. Methods should have verb or verb phrase names
11. Be simple. Be Direct. Say what you mean, mean what you say.
12. Don't use the same word for 2 purposes
13.

## Functions rules
1. Small.
2. Do one thing.
3. Use descriptive names.
4. Prefer fewer arguments.
5. Have no side effects.
6. Don't use flag arguments. Split method into several independent methods that can be called from the client without the flag.
7. Smaller than 20 lines long
8. The Stepdown rule => function -> next level of abstraction


## ErrorHandling
1. Specify where the error in print
2. Don't use a single variable
3. 

## If statements
1. 


## Comments rules
1. Always try to explain yourself in code.
2. Don't be redundant.
3. Don't add obvious noise.
4. Don't use closing brace comments.
5. Don't comment out code. Just remove.
6. Use as explanation of intent.
7. Use as clarification of code.
8. Use as warning of consequences.

## Source code structure
1. Separate concepts vertically.
2. Related code should appear vertically dense.
3. Declare variables close to their usage.
4. Dependent functions should be close.
5. Similar functions should be close.
6. Place functions in the downward direction.
7. Keep lines short.
8. Don't use horizontal alignment.
9. Use white space to associate related things and disassociate weakly related.
10. Don't break indentation.

## Objects and data structures
1. Hide internal structure.
2. Prefer data structures.
3. Avoid hybrids structures (half object and half data).
4. Should be small.
5. Do one thing.
6. Small number of instance variables.
7. Base class should know nothing about their derivatives.
8. Better to have many functions than to pass some code into a function to select a behavior.
9. Prefer non-static methods to static methods.

## Tests
1. One assert per test.
2. Readable.
3. Fast.
4. Independent.
5. Repeatable.

## Code smells
1. Rigidity. The software is difficult to change. A small change causes a cascade of subsequent changes.
2. Fragility. The software breaks in many places due to a single change.
3. Immobility. You cannot reuse parts of the code in other projects because of involved risks and high effort.
4. Needless Complexity.
5. Needless Repetition.
6. Opacity. The code is hard to understand.








# Clean Code

Here are some general principles for writing highly usable, functional, reliable, fast, and scalable code:

1. **Clear and Understandable:** The code should be written in a way that's easy for others to understand. This includes using clear variable and function names, and including comments to explain complex sections of code.

2. **Modular and Reusable:** Code should be broken down into small, modular functions and classes that each perform a single task. This makes the code more understandable, and also allows for code reuse.

3. **Robust Error Handling:** The code should be able to handle all potential errors gracefully, and should never crash unexpectedly. This includes checking for invalid input, catching exceptions, and providing useful error messages.

4. **Type Handling:** Whenever possible, the code should enforce and check types to prevent type-related errors. This can be done through the use of type hints in languages like Python, or through explicit type checks.

5. **Logging:** The code should include extensive logging to make it easier to debug and understand what the code is doing. This includes logging any errors that occur, as well as important events or state changes.

6. **Performance:** The code should be optimized for performance, avoiding unnecessary computation and using efficient algorithms and data structures. This includes profiling the code to identify and optimize performance bottlenecks.

7. **Scalability:** The code should be designed to scale well as the size of the input data or the number of users increases. This includes using scalable algorithms and data structures, and designing the code to work well in a distributed or parallel computing environment if necessary.

8. **Testing:** The code should include comprehensive tests to ensure that it works correctly. This includes unit tests for individual functions and classes, as well as integration tests to ensure that the different parts of the code work well together.

9. **Version Control:** The code should be stored in a version control system like Git, which allows for tracking changes, collaborating with others, and rolling back to a previous state if necessary.

10. **Documentation:** The codebase should be well-documented, both in terms of comments within the code and external documentation that explains how to use and contribute to the code.

11. **Continuous Integration/Continuous Deployment (CI/CD):** Implement CI/CD pipelines for automatic testing and deployment. This ensures that any new changes do not break existing functionality and that the latest version of the application is always available for deployment.

# Examples
1. **Clear and Understandable:** Use meaningful variable and function names. Include comments when necessary.

    ```python
    # Good example
    def calculate_average(numbers: List[int]) -> float:
        """Calculate and return the average of a list of numbers."""
        total = sum(numbers)
        count = len(numbers)
        return total / count
    ```
   
   For file and folder names, use descriptive names that relate to their function in your program. For example, a file that contains functions for handling user input might be named `user_input.py`.

2. **Modular and Reusable:** Write functions for tasks that you perform over and over.

    ```python
    def greet_user(name: str):
        """Print a greeting to the user."""
        print(f"Hello, {name}!")
    ```

    For folder structure, group related files in the same directory. For example, all test files could be in a `tests` directory.

3. **Robust Error Handling:** Use try/except blocks to catch and handle errors.

    ```python
    def divide_numbers(numerator: float, denominator: float) -> float:
        """Divide two numbers and handle division by zero."""
        try:
            return numerator / denominator
        except ZeroDivisionError:
            print("Error: Division by zero.")
            return None
    ```
   
4. **Type Handling:** Use type hints to specify the type of function arguments and return values.

    ```python
    def greet_user(name: str) -> None:
        """Greet the user."""
        print(f"Hello, {name}!")
    ```

5. **Logging:** Use the `logging` module to log events.

    ```python
    import logging

    logging.basicConfig(level=logging.INFO)

    def divide_numbers(numerator: float, denominator: float) -> float:
        """Divide two numbers and log if division by zero occurs."""
        try:
            return numerator / denominator
        except ZeroDivisionError:
            logging.error("Attempted division by zero.")
            return None
    ```

6. **Performance:** Use built-in functions and data types for better performance.

    ```python
    # Using a set to check for membership is faster than using a list
    numbers_set = set(numbers)
    if target in numbers_set:
        print(f"{target} is in the set of numbers.")
    ```
   
7. **Scalability:** For scalability, an example might involve using a load balancer or dividing tasks among different workers or threads. This is more of a system design consideration than a single piece of code.

8. **Testing:** Write tests for your functions.

    ```python
    def test_calculate_average():
        assert calculate_average([1, 2, 3, 4]) == 2.5
    ```

    For tests, you could have a separate `tests` directory. Inside this directory, each test file could be named `test_<filename>.py` where `<filename>` is the name of the file being tested.

9. **Version Control:** This point refers to using tools like Git for version control. A simple example would be committing changes to a repository:

    ```bash
    git add .
    git commit -m "Add function to calculate average"
    git push
    ```

10. **Documentation:** Write docstrings for your functions.

    ```python
    def calculate_average(numbers: List[int]) -> float:
        """Calculate and return the average of a list of numbers."""
        ...
    ```

    Documentation might be kept in a `docs` directory, with separate files for different topics.

11. **Continuous Integration/Continuous Deployment (CI/CD):** This is typically handled by a system like Jenkins, GitHub Actions, or GitLab CI/CD. It involves creating a script or configuration file that tells the CI/CD system how to build, test, and deploy your code. For example, a `.github/workflows/main.yml` file for a GitHub Actions workflow.

Remember, consistency in your naming conventions and organization is key. Having a standard and sticking to it will make your codebase easier to navigate and understand.