# Test Server

This is the testing server for swarms. You can use this server to mock rest calls. You can replace any agent that has rest calls with this. Even if the agent has a client (SDK), in the end of the day it may expose a REST API for interaction.

## Requirements

- [Pip](https://pypi.org/project/pip/)
- [venv](https://docs.python.org/3/library/venv.html)

## Get started

Make sure you are in the `test-server` folder.

- Create a virtual environment

```bash
$ python -m venv venv
```

If you are using Windows, use `c:\>python -m venv c:\path\to\myenv`.

- Active it

```bash
$ source venv/bin/activate
```

or `C:\> <venv>\Scripts\activate.bat` where `<venv>` is the same as `c:\path\to\myenv`.

- Install dependencies

```bash
$ pip install -r requirements.txt
```

- Run it

```bash
$ fastapi dev main.py
```

You can then access the documentation by going to [ http://localhost:8000/docs](http://localhost:8000/docs).
