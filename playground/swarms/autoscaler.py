from swarms import AutoScaler

auto_scaler = AutoScaler()
auto_scaler.start()

for i in range(100):
    auto_scaler.add_task(f"Task {i}")
