from swarms import Agent, Swarm
from swarms.utils.language_config import Language

def main():
    # ایجاد عامل‌ها با زبان فارسی
    agent1 = Agent(
        name="تحلیل‌گر",
        task="تحلیل داده‌های فروش",
        language=Language.PERSIAN
    )

    agent2 = Agent(
        name="گزارش‌گر",
        task="تولید گزارش تحلیلی",
        language=Language.PERSIAN
    )

    # ایجاد گروه با زبان فارسی
    swarm = Swarm(
        agents=[agent1, agent2],
        language=Language.PERSIAN
    )

    # اجرای وظیفه
    task = """
    لطفاً داده‌های فروش سه ماه گذشته را تحلیل کنید و یک گزارش تحلیلی تهیه کنید.
    در گزارش موارد زیر را بررسی کنید:
    1. روند فروش ماهانه
    2. محصولات پرفروش
    3. پیشنهادات برای بهبود فروش
    """

    try:
        results = swarm.run(task)
        print("\nنتایج:")
        for agent_name, result in results.items():
            print(f"\n{agent_name}:")
            print(result)
    except Exception as e:
        print(f"خطا در اجرای وظیفه: {str(e)}")

if __name__ == "__main__":
    main() 