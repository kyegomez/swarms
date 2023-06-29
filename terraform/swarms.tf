provider "aws" {
  region = "us-west-2"
}

data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"]
}

resource "aws_key_pair" "deployer" {
  key_name   = "deployer-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "aws_security_group" "swarm-sg" {
  name        = "swarm-sg"
  description = "Security group for Swarm app"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_launch_configuration" "swarm" {
  name          = "swarm-configuration"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = "g4dn.xlarge"
  key_name      = aws_key_pair.deployer.key_name

  security_groups = [aws_security_group.swarm-sg.id]

  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y docker.io
              sudo docker pull your_docker_image_name
              sudo docker run -d -p 8000:8000 your_docker_image_name
              EOF

  root_block_device {
    volume_type = "gp2"
    volume_size = 30  # size in GBs
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "swarm" {
  name_prefix          = "swarm-asg"
  max_size             = 5
  min_size             = 1
  desired_capacity     = 1
  launch_configuration = aws_launch_configuration.swarm.id

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_elb" "swarm" {
  name    = "swarm-elb"
  subnets = ["subnet-id1", "subnet-id2"]

  listener {
    instance_port     = 8000
    instance_protocol = "http"
    lb_port           = 80
    lb_protocol       = "http"
  }

  health_check {
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 3
    target              = "HTTP:8000/"
    interval            = 30
  }

  instances = [aws_instance.swarm.id]

  cross_zone_load_balancing   = true
  idle_timeout                = 400
  connection_draining         = true
  connection_draining_timeout = 400
}
