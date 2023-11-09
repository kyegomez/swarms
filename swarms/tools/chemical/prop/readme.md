# Chemical Properties

Contributor: [Zheni Zeng](https://github.com/Ellenzzn)

## Tool Description
The tool, "Chemical Property Plugin," provides the ability to lookup a chemical's properties by querying a chemical knowledge base. The tool accepts the input in a JSON format, like {'input': 'some input'} and guides you to ask questions and search step by step.

### Tool Specifications

- **Name**: Chemical Property Plugin
- **Purpose**: Plugin for looking up a chemical's property using a chemical knowledge base
- **Logo**: ![Chemical Property Plugin Logo](https://your-app-url.com/.well-known/logo.png)
- **Contact Email**: hello@contact.com
- **Legal Information**: [Legal Information](hello@legal.com)

### Core Functionality

1. `get_name`

    This method accepts a Compound ID (CID) and returns the top 3 synonyms for the queried compound. 

2. `get_allname`

    This method accepts a Compound ID (CID) and returns all the possible synonyms for the queried compound. Be aware that the number of returned names can be large, so use this function with caution.

3. `get_id_by_struct`

    This method accepts a SMILES formula and returns the ID of the queried compound. This method should be used only if the SMILES formula is provided or retrieved in the previous step. 

4. `get_id`

    This method accepts a compound name and returns the ID of the queried compound. If the name cannot be precisely matched, it will return the possible names. 

5. `get_prop`

    This method accepts a Compound ID (CID) and returns the properties of the queried compound.

The tool is made possible through the use of the ChemicalPropAPI, which interacts with a chemical knowledge base.