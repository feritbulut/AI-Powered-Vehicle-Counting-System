using AracSayar.Data;

namespace AracSayar
{
    public partial class App : Application
    {
        public static MeasurementDatabase Database { get; private set; }
        public App()
        {
            InitializeComponent();
            
            var dbPath = Path.Combine(
                FileSystem.AppDataDirectory, "Measurements.db");

            Database = new MeasurementDatabase(dbPath);

            MainPage = new NavigationPage(new MainPage());
        }

    }
}