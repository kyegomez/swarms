import inspect
import pkgutil


class ModelRegistry:
    """
    A registry for storing and querying models.

    Attributes:
        models (dict): A dictionary of model names and corresponding model classes.

    Methods:
        __init__(): Initializes the ModelRegistry object and retrieves all available models.
        _get_all_models(): Retrieves all available models from the models package.
        query(text): Queries the models based on the given text and returns a dictionary of matching models.
    """

    def __init__(self):
        self.models = self._get_all_models()

    def _get_all_models(self):
        """
        Retrieves all available models from the models package.

        Returns:
            dict: A dictionary of model names and corresponding model classes.
        """
        models = {}
        for importer, modname, ispkg in pkgutil.iter_modules(
            models.__path__
        ):
            module = importer.find_module(modname).load_module(
                modname
            )
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    models[name] = obj
        return models

    def query(self, text):
        """
        Queries the models based on the given text and returns a dictionary of matching models.

        Args:
            text (str): The text to search for in the model names.

        Returns:
            dict: A dictionary of matching model names and corresponding model classes.
        """
        return {
            name: model
            for name, model in self.models.items()
            if text in name
        }

    def run_model(
        self, model_name: str, task: str, img: str, *args, **kwargs
    ):
        """
        Runs the specified model for the given task and image.

        Args:
            model_name (str): The name of the model to run.
            task (str): The task to perform using the model.
            img (str): The image to process.
            *args: Additional positional arguments to pass to the model's run method.
            **kwargs: Additional keyword arguments to pass to the model's run method.

        Returns:
            The result of running the model.

        Raises:
            ValueError: If the specified model is not found in the model registry.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        # Get the model
        model = self.models[model_name]

        # Run the model
        return model.run(task, img, *args, **kwargs)
