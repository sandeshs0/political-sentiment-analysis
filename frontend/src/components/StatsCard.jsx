const StatsCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  color = "primary",
}) => {
  const colorClasses = {
    primary: "text-primary-600 bg-primary-50",
    success: "text-success-600 bg-success-50",
    danger: "text-danger-600 bg-danger-50",
    warning: "text-warning-600 bg-warning-50",
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 card-hover">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
        </div>
        {Icon && (
          <div className={`p-3 rounded-full ${colorClasses[color]}`}>
            <Icon className="h-6 w-6" />
          </div>
        )}
      </div>
    </div>
  );
};

export default StatsCard;
