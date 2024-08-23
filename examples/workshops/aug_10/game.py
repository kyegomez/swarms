import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Psychedelic Pulse")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Player
player_radius = 10
player_x = WIDTH // 2
player_y = HEIGHT - 50

# Goal
goal_radius = 20
goal_x = WIDTH // 2
goal_y = 50


# Obstacles
class PsychedelicShape:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(100, HEIGHT - 100)
        self.radius = random.randint(20, 60)
        self.color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        self.pulse_speed = random.uniform(0.05, 0.2)
        self.move_speed = random.uniform(1, 3)
        self.direction = random.choice([-1, 1])

    def update(self):
        self.radius = (
            abs(math.sin(pygame.time.get_ticks() * self.pulse_speed)) * 40
            + 20
        )
        self.x += self.move_speed * self.direction
        if self.x < 0 or self.x > WIDTH:
            self.direction *= -1

    def draw(self):
        pygame.draw.circle(
            screen,
            self.color,
            (int(self.x), int(self.y)),
            int(self.radius),
        )


# Create obstacles
obstacles = [PsychedelicShape() for _ in range(10)]

# Game loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move player
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > player_radius:
        player_x -= 5
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_radius:
        player_x += 5
    if keys[pygame.K_UP] and player_y > player_radius:
        player_y -= 5
    if keys[pygame.K_DOWN] and player_y < HEIGHT - player_radius:
        player_y += 5

    # Update obstacles
    for obstacle in obstacles:
        obstacle.update()

    # Check for collisions
    for obstacle in obstacles:
        distance = math.sqrt(
            (player_x - obstacle.x) ** 2 + (player_y - obstacle.y) ** 2
        )
        if distance < player_radius + obstacle.radius:
            player_x = WIDTH // 2
            player_y = HEIGHT - 50

    # Check for goal
    if (
        math.sqrt((player_x - goal_x) ** 2 + (player_y - goal_y) ** 2)
        < player_radius + goal_radius
    ):
        print("You win!")
        running = False

    # Draw everything
    screen.fill(BLACK)
    for obstacle in obstacles:
        obstacle.draw()
    pygame.draw.circle(
        screen, WHITE, (int(player_x), int(player_y)), player_radius
    )
    pygame.draw.circle(
        screen, (255, 215, 0), (goal_x, goal_y), goal_radius
    )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
