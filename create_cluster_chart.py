import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("static/student_clusters.csv")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='G3', y='absences', hue='cluster', data=data, palette='viridis')
plt.title('Распределение студентов по итоговой оценке и пропускам')
plt.xlabel('Итоговая оценка (G3)')
plt.ylabel('Количество пропусков')
plt.legend(title='Кластер')
plt.savefig("static/cluster_chart.png")
plt.close()
