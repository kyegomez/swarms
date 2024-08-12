import pygame
import random
import math

# Initialize Pygame and mixer
pygame.init()
pygame.mixer.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Psychedelic Soundscape Explorer")

# Colors
BLACK = (0, 0, 0)

# Player
player_pos = [WIDTH // 2, HEIGHT // 2]
player_radius = 20

# Sound zones
sound_zones = []
for _ in range(5):
    sound_zones.append(
        [
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
            random.randint(50, 150),
        ]
    )

# Create sounds
sounds = [pygame.mixer.Sound(f"sound{i}.wav") for i in range(1, 6)]

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move player
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_pos[0] -= 5
    if keys[pygame.K_RIGHT]:
        player_pos[0] += 5
    if keys[pygame.K_UP]:
        player_pos[1] -= 5
    if keys[pygame.K_DOWN]:
        player_pos[1] += 5

    # Clear the screen
    screen.fill(BLACK)

    # Draw and play sounds
    for i, (x, y, radius) in enumerate(sound_zones):
        distance = math.sqrt(
            (player_pos[0] - x) ** 2 + (player_pos[1] - y) ** 2
        )
        if distance < radius:
            intensity = 1 - (distance / radius)
            sounds[i].set_volume(intensity)
            sounds[i].play(-1)

            # Create trippy color based on distance and sound
            r = int(255 * math.sin(intensity * math.pi / 2))
            g = int(255 * math.cos(intensity * math.pi / 2))
            b = int(255 * (1 - intensity))

            pygame.draw.circle(
                screen, (r, g, b), (x, y), int(radius * intensity), 2
            )
        else:
            sounds[i].stop()

    # Draw player
    pygame.draw.circle(screen, (255, 255, 255), player_pos, player_radius)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
