from setuptools import setup, find_packages

# from setuptools.command.install import install

# class PostInstallCommand(install):
#     """Post-installation for installation mode."""
#     def run(self):
#         install.run(self)
#         # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
#         import subprocess
#         subprocess.check_call(["pip", "install", "git+https://github.com/IDEA-Research/GroundingDINO.git"])
#         subprocess.check_call(["pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"])


setup(
  name = 'swarms',
  packages = find_packages(exclude=[]),
  version = '0.2.8',
  license='MIT',
  description = 'Swarms - Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/swarms',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers',
    "Prompt Engineering"
  ],
  install_requires=[
        "transformers",
        "openai",
        "langchain",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "asyncio",
        "nest_asyncio",
        "bs4",
        "playwright",
        "duckduckgo_search",
        "faiss-cpu",
        "wget==3.2",
        "accelerate",
        "addict",
        "albumentations",
        "basicsr",
        "controlnet-aux",
        "diffusers",
        "einops",
        "gradio",
        "imageio",
        "imageio-ffmpeg",
        "invisible-watermark",
        "kornia",
        "numpy",
        "omegaconf",
        "open_clip_torch",
        "opencv-python",
        "prettytable",
        "safetensors",
        "streamlit",
        "test-tube",
        "timm",
        "torchmetrics",
        "webdataset",
        "yapf",
        # 'GroundingDINO'
        "wolframalpha",
        "wikipedia",
        "httpx",
        "ggl",
        "gradio_tools",
        "arxiv",
        "google-api-python-client",
        "google-auth-oauthlib",
        "google-auth-httplib2",
        "beautifulsoup4",
        "O365",
        "pytube",
        "pydub"
    ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)