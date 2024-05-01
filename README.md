# Building block: Portfolio optimization

This building block is included in the [TNO Quantum Python Toolbox](https://ci.tno.nl/gitlab/quantum/quantum-applications/quantum-toolbox/quantum-toolbox).


See also the related [examples repository](https://ci.tno.nl/gitlab/quantum/quantum-applications/quantum-toolbox/toolbox-examples).

## Install using access token

1. Generate a personal access token with `read_api` scope. Instructions are found [here](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html).
1. Make sure pip is up to date:
   ```commandline
   pip install --upgrade pip
   ```
1. Install:
   ```commandline
   pip install tno.quantum.problems.portfolio_optimization --index-url https://__token__:<your_personal_token>@ci.tno.nl/gitlab/api/v4/groups/4737/-/packages/pypi/simple
   ```

## Install in dev mode using access token

1. Generate a personal access token with `read_api` scope. Instructions are found [here](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html).
1. Clone the repository:
   ```commandline
   git clone git@ci.tno.nl/gitlab/quantum/quantum-applications/quantum-toolbox/microlibs/problems/microlibs/portfolio-optimization.git.git
   ```
1. Make sure pip is up to date:
   ```commandline
   pip install --upgrade pip
   ```
1. Install:
   ```commandline
   pip install -e .[dev,tests] --index-url https://__token__:<your_personal_token>@ci.tno.nl/gitlab/api/v4/groups/4737/-/packages/pypi/simple
   ```

## Building the docs
To generate the documentation locally, install the TNO [dev-utils](https://ci.tno.nl/gitlab/quantum/cicd/dev_utils) and run `tno-make-docs`. 

The HTML documentation will be placed in a directory called `docs/`.
This folder includes a file `index.html`, which can be opened in a browser to view the documentation.

## (End)use limitations
The content of this software may solely be used for applications that comply with international export control laws.