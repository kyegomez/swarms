from swarms.agents.agent_judge import AgentJudge


judge = AgentJudge(model_name="gpt-4o-mini", max_loops=1)


outputs = [
    "1. Agent CalculusMaster: After careful evaluation, I have computed the integral of the polynomial function. The result is ∫(x^2 + 3x + 2)dx = (1/3)x^3 + (3/2)x^2 + 5, where I applied the power rule for integration and added the constant of integration.",
    "2. Agent DerivativeDynamo: In my analysis of the function sin(x), I have derived it with respect to x. The derivative is d/dx (sin(x)) = cos(x). However, I must note that the additional term '+ 2' is not applicable in this context as it does not pertain to the derivative of sin(x).",
    "3. Agent LimitWizard: Upon evaluating the limit as x approaches 0 for the function (sin(x)/x), I conclude that lim (x -> 0) (sin(x)/x) = 1. The additional '+ 3' is incorrect and should be disregarded as it does not relate to the limit calculation.",
    "4. Agent IntegralGenius: I have computed the integral of the exponential function e^x. The result is ∫(e^x)dx = e^x + C, where C is the constant of integration. The extra '+ 1' is unnecessary and does not belong in the final expression.",
    "5. Agent FunctionFreak: Analyzing the cubic function f(x) = x^3 - 3x + 2, I determined that it has a maximum at x = 1. However, the additional '+ 2' is misleading and should not be included in the maximum value statement.",
]

print(judge.run(outputs))
