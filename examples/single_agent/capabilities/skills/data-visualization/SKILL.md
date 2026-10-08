---
name: data-visualization
description: Create effective data visualizations using best practices for clarity, accuracy, and visual communication of insights
---

# Data Visualization Skill

When creating data visualizations, follow these principles to ensure clear and effective communication:

## Core Principles

### 1. Choose the Right Chart Type
- **Line Charts**: Trends over time, continuous data
- **Bar Charts**: Comparing categories, discrete data
- **Scatter Plots**: Relationships between variables, correlations
- **Pie Charts**: Parts of a whole (use sparingly, max 5-6 segments)
- **Heatmaps**: Patterns in large datasets, correlations
- **Box Plots**: Distribution statistics, outlier detection

### 2. Design Guidelines

**Clarity**
- Use clear, descriptive titles and labels
- Include units of measurement
- Add a legend when multiple series are present
- Ensure adequate contrast and readability

**Accuracy**
- Start y-axis at zero for bar charts (unless good reason)
- Use consistent scales across related charts
- Avoid distorting data through inappropriate scaling
- Label data points when precision matters

**Simplicity**
- Remove chart junk and unnecessary decorations
- Use color purposefully, not decoratively
- Limit the number of colors (5-7 max)
- Ensure accessibility (colorblind-friendly palettes)

### 3. Color Best Practices
- **Sequential**: Use for ordered data (light to dark)
- **Diverging**: Use for data with a meaningful midpoint
- **Categorical**: Use for unordered categories
- **Highlight**: Use accent colors to draw attention
- Test accessibility with colorblind simulators

### 4. Storytelling with Data
- Lead with the insight, not the data
- Use annotations to highlight key findings
- Arrange charts in logical flow
- Provide context and comparisons
- Include data sources and timestamp

## Visualization Workflow

1. **Understand the Data**
   - Explore data structure and distributions
   - Identify key variables and relationships
   - Determine the message to communicate

2. **Select Visualization Type**
   - Match chart type to data characteristics
   - Consider audience and use case
   - Plan for interactivity if needed

3. **Design the Visualization**
   - Create initial draft
   - Apply design principles
   - Optimize for clarity and impact

4. **Refine and Validate**
   - Get feedback from stakeholders
   - Test on target audience
   - Iterate based on feedback
   - Verify accuracy

## Common Mistakes to Avoid

- Using 3D charts unnecessarily (adds confusion)
- Too many colors or visual elements
- Missing or unclear axis labels
- Truncated y-axis to exaggerate differences
- Using pie charts for more than 5-6 categories
- Poor color choices (rainbow colors for sequential data)

## Tools and Libraries

Recommend appropriate tools based on needs:
- **Python**: matplotlib, seaborn, plotly, altair
- **R**: ggplot2, plotly
- **JavaScript**: D3.js, Chart.js, Highcharts
- **BI Tools**: Tableau, Power BI, Looker

## Example Use Cases

- **Dashboard Design**: "Create an executive dashboard for sales metrics"
- **Exploratory Analysis**: "Visualize patterns in customer behavior data"
- **Report Charts**: "Generate publication-ready charts for annual report"
