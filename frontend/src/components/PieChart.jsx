import { ArcElement, Chart as ChartJS, Legend, Tooltip } from "chart.js";
import { Pie } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip, Legend);

const PieChart = ({ data, title, colors }) => {
  const chartData = {
    labels: Object.keys(data),
    datasets: [
      {
        data: Object.values(data).map((val) => (val * 100).toFixed(1)),
        backgroundColor: colors || [
          "#22c55e", // Green
          "#ef4444", // Red
          "#6b7280", // Gray
          "#3b82f6", // Blue
          "#f59e0b", // Yellow
          "#ec4899", // Pink
        ],
        borderWidth: 2,
        borderColor: "#ffffff",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: {
          padding: 20,
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            return `${context.label}: ${context.parsed}%`;
          },
        },
      },
    },
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
      <div className="h-64">
        <Pie data={chartData} options={options} />
      </div>
    </div>
  );
};

export default PieChart;
