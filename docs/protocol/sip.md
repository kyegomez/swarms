# Swarms Improvement Proposal (SIP) Guidelines

A simplified process for proposing new functionality and enhancements to the Swarms framework.

## What is a SIP?

A **Swarms Improvement Proposal (SIP)** is a design document that describes a new feature, enhancement, or change to the Swarms framework. SIPs serve as the primary mechanism for proposing significant changes, collecting community feedback, and documenting design decisions.

The SIP author is responsible for building consensus within the community and documenting the proposal clearly and concisely.

## When to Submit a SIP

Consider submitting a SIP for:

- **New Agent Types or Behaviors**: Adding new agent architectures, swarm patterns, or coordination mechanisms
- **Core Framework Changes**: Modifications to the Swarms API, core classes, or fundamental behaviors
- **New Integrations**: Adding support for new LLM providers, tools, or external services
- **Breaking Changes**: Any change that affects backward compatibility
- **Complex Features**: Multi-component features that require community discussion and design review

For simple bug fixes, minor enhancements, or straightforward additions, use regular GitHub issues and pull requests instead.

## SIP Types

**Standard SIP**: Describes a new feature or change to the Swarms framework
**Process SIP**: Describes changes to development processes, governance, or community guidelines
**Informational SIP**: Provides information or guidelines to the community without proposing changes

## Submitting a SIP

1. **Discuss First**: Post your idea in [GitHub Discussions](https://github.com/kyegomez/swarms/discussions) to gauge community interest
2. **Create Issue**: Submit your SIP as a GitHub Issue with the `SIP` and `proposal` labels
3. **Follow Format**: Use the SIP template format below
4. **Engage Community**: Respond to feedback and iterate on your proposal

## SIP Format

### Required Sections

#### **SIP Header**
```
Title: [Descriptive title]
Author: [Your name and contact]
Type: [Standard/Process/Informational]
Status: Proposal
Created: [Date]
```

#### **Abstract** (200 words max)
A brief summary of what you're proposing and why.

#### **Motivation**
- What problem does this solve?
- Why can't the current framework handle this?
- What are the benefits to the Swarms ecosystem?

#### **Specification**
- Detailed technical description
- API changes or new interfaces
- Code examples showing usage
- Integration points with existing framework

#### **Implementation Plan**
- High-level implementation approach
- Breaking changes (if any)
- Migration path for existing users
- Testing strategy

#### **Alternatives Considered**
- Other approaches you evaluated
- Why you chose this solution
- Trade-offs and limitations

### Optional Sections

#### **Reference Implementation**
Link to prototype code or proof-of-concept (can be added later)

#### **Security Considerations**
Any security implications or requirements

## SIP Workflow

```
Proposal → Draft → Review → Accepted/Rejected → Final
```

1. **Proposal**: Initial submission as GitHub Issue
2. **Draft**: Maintainer assigns SIP number and `draft` label
3. **Review**: Community and maintainer review period
4. **Decision**: Accepted, rejected, or needs revision
5. **Final**: Implementation completed and merged

## SIP Status

- **Proposal**: Newly submitted, awaiting initial review
- **Draft**: Under active discussion and refinement
- **Review**: Formal review by maintainers
- **Accepted**: Approved for implementation
- **Rejected**: Not accepted (with reasons)
- **Final**: Implementation completed and merged
- **Withdrawn**: Author withdrew the proposal

## Review Process

- SIPs are reviewed during regular maintainer meetings
- Community feedback is collected via GitHub comments
- Acceptance requires:
  - Clear benefit to the Swarms ecosystem
  - Technical feasibility
  - Community support
  - Working prototype (for complex features)

## Getting Help

- **Discussions**: Use [GitHub Discussions](https://github.com/kyegomez/swarms/discussions) for questions
- **Documentation**: Check [docs.swarms.world](https://docs.swarms.world) for framework details
- **Examples**: Look at existing SIPs for reference

## SIP Template

When creating your SIP, copy this template:

```markdown
# SIP-XXX: [Title]

**Author**: [Your name] <[email]>
**Type**: Standard
**Status**: Proposal
**Created**: [Date]

## Abstract

[Brief 200-word summary]

## Motivation

[Why is this needed? What problem does it solve?]

## Specification

[Detailed technical description with code examples]

## Implementation Plan

[How will this be built? Any breaking changes?]

## Alternatives Considered

[Other approaches and why you chose this one]

## Reference Implementation

[Link to prototype code if available]
```

---

**Note**: This process is designed to be lightweight while ensuring important changes get proper community review. For questions about whether your idea needs a SIP, start a discussion in the GitHub Discussions forum.