import time
from game.game_core import Game
import csv

class BenchmarkRunner:
    def __init__(self,num_runs=10):
        self.num_runs = num_runs
        self.results = {}
    def run(self,dodgeMethod, save_csv = True, csv_filename="benchmark_result.csv"):
        all_data = []
        for name, dodgeMethod in dodgeMethod.items():
            print(f"Running:{name}")
            times= []
            for i in range(self.num_runs):
                bot = dodgeMethod()
                start = time.perf_counter()
                bot.reset_action()
                end = time.perf_counter()
                duration = end -start
                times.append(duration)
                all_data.append({"algorithim": name, "run": i + 1,"time": duration})
            
            min_time = min(times)
            max_time=max(times)
            avg_time=sum(times)/len(times)
            
            self.results[name]={
                'runs': times,
                'min_time': min_time,
                'max_time': max_time,
                'avg_time': avg_time,

            }
            for i in range(self.num_runs):
                all_data[-self.num_runs+i]["min_time"] = min_time
                all_data[-self.num_runs+i]["max_time"] = max_time
                all_data[-self.num_runs+i]["avg_time"] = avg_time

        if save_csv:
            keys =["algorithim","run","time"]
            with open(csv_filename,"w",newline='') as f:
                writer = csv.DictWriter(f, fieldnames =keys+['min_time','max_time','avg_time'])
                writer.writeheader()
                writer.writerows(all_data)
