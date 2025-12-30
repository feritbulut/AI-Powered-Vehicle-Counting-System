using AracSayar.Models;
using System.Text.Json;

namespace AracSayar
{
    public partial class MainPage : ContentPage
    {
        private readonly HttpClient _httpClient = new();
        private int _startValue = 0;
        private bool _isMeasuring = false;
        private DateTime _measuremenStartTime;
        private DateTime _measuremenEndTime;
        private int _lastMeasuredCount;

        private readonly List<MeasurementResult> _measurements = new();

        private const string ApiUrl = "https://ai-vehicle-detection-backend-api.onrender.com/api/stats";
        public MainPage()
        {
            InitializeComponent();
        }

        private async void OnStartMeasurementClicked(object sender,EventArgs e)
        {
            if (!_isMeasuring)
            {
                var currentTotal = await GetTotalVehiclesAsync();
                if (currentTotal == null)
                {
                    return;
                }

                StartStopButton.BackgroundColor = Color.FromArgb("#e74c3c");

                _startValue = currentTotal.Value;
                CounterLabel.Text = "0";
                _isMeasuring = true;
                _measuremenStartTime = DateTime.Now;
                StartStopButton.Text = "Ölçümü Durdur";

                StartPolling();
            }
            else
            {
                _measuremenEndTime = DateTime.Now;
                _lastMeasuredCount = int.Parse(CounterLabel.Text);

                var duration = _measuremenEndTime - _measuremenStartTime;

                _isMeasuring = false;
                StartStopButton.Text = "Ölçümü Başlat";
                _cts.Cancel();

                StartStopButton.BackgroundColor = Color.FromArgb("#27AE60");

                var result = new MeasurementResult
                {
                    Date = _measuremenStartTime.Date,
                    StartTime = _measuremenStartTime,
                    EndTime = _measuremenEndTime,
                    DurationMinutes = duration.TotalMinutes,
                    VehicleCount = _lastMeasuredCount
                };

                await App.Database.SaveMeasurementAsync(result);

                await DisplayAlert(
                    "Ölçüm Kaydedildi",
                    $"Tarih: {result.Date:dd.MM.yyyy}\n" +
                    $"Başlangıç: {result.StartTime:HH:mm:ss}\n" +
                    $"Bitiş: {result.EndTime:HH:mm:ss}\n" +
                    $"Süre: {result.DurationMinutes:F2} dk\n" +
                    $"Araç Sayısı: {result.VehicleCount}",
                    "Tamam");

            }
        }

        private CancellationTokenSource _cts;

        private async void StartPolling()
        {
            _cts = new CancellationTokenSource();
            var timer = new PeriodicTimer(TimeSpan.FromSeconds(1));
            try
            {
                while (_isMeasuring && await timer.WaitForNextTickAsync(_cts.Token))
                {
                    await UpdateCounterAsync();
                }
            }
            catch (OperationCanceledException)
            {
                
            }
        }


        //private void StartPolling()
        //{
        //    Device.StartTimer(TimeSpan.FromSeconds(1), () =>
        //    {
        //        _ = UpdateCounterAsync();
        //        return _isMeasuring;
        //    });
        //}

        private async Task UpdateCounterAsync()
        {
            var currentTotal = await GetTotalVehiclesAsync();
            if (currentTotal == null)
            {
                return;
            }

            var difference = currentTotal.Value - _startValue;
            if (difference < 0)
            {
                difference = 0;
            }

            MainThread.BeginInvokeOnMainThread(() =>
            {
                CounterLabel.Text = difference.ToString();
            });
        }


        private async Task<int?> GetTotalVehiclesAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync(ApiUrl);
                if (!response.IsSuccessStatusCode)
                    return null;

                var json = await response.Content.ReadAsStringAsync();
                var data = JsonSerializer.Deserialize<StatsResponse>(json);

                return data?.total_vehicles;
            }
            catch
            {
                return null;
            }
        }

        private async void OnShowMeasurementsClicked(object sender, EventArgs e)
        {
            await Navigation.PushAsync(new MeasurementsPage());
        }


    }

}
