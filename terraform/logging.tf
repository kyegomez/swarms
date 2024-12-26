provider "aws" {
  region = "us-west-2"
}

resource "aws_cloudwatch_log_group" "swarms_log_group" {
  name              = "swarms-log-group"
  retention_in_days = 14
}

resource "aws_iam_role" "swarms_logging_role" {
  name = "swarms-logging-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_policy" "swarms_logging_policy" {
  name        = "swarms-logging-policy"
  description = "Policy for allowing swarms to create and manage CloudWatch log groups"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:PutRetentionPolicy",
          "logs:TagLogGroup"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "swarms_logging_role_attachment" {
  role       = aws_iam_role.swarms_logging_role.name
  policy_arn = aws_iam_policy.swarms_logging_policy.arn
}
