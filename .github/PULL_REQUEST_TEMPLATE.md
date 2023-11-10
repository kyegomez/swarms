Thank you for contributing to Swarms!

Replace this comment with:
  - Description: a description of the change, 
  - Issue: the issue # it fixes (if applicable),
  - Dependencies: any dependencies required for this change,
  - Tag maintainer: for a quicker response, tag the relevant maintainer (see below),
  - Twitter handle: we announce bigger features on Twitter. If your PR gets announced and you'd like a mention, we'll gladly shout you out!

Please make sure your PR is passing linting and testing before submitting. Run `make format`, `make lint` and `make test` to check this locally.

See contribution guidelines for more information on how to write/run tests, lint, etc: 
https://github.com/kyegomez/swarms/blob/master/CONTRIBUTING.md

If you're adding a new integration, please include:
  1. a test for the integration, preferably unit tests that do not rely on network access,
  2. an example notebook showing its use.


Maintainer responsibilities:
  - General / Misc / if you don't know who to tag: kye@apac.ai
  - DataLoaders / VectorStores / Retrievers: kye@apac.ai
  - swarms.models: kye@apac.ai
  - swarms.memory: kye@apac.ai
  - swarms.structures: kye@apac.ai

If no one reviews your PR within a few days, feel free to email Kye at kye@apac.ai

See contribution guidelines for more information on how to write/run tests, lint, etc: https://github.com/kyegomez/swarms