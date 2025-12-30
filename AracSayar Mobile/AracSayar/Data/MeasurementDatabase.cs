using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SQLite;
using AracSayar.Models;

namespace AracSayar.Data
{
    public class MeasurementDatabase
    {
        private readonly SQLiteAsyncConnection _database;

        public MeasurementDatabase(string dbPath)
        {
            _database = new SQLiteAsyncConnection(dbPath);
            _database.CreateTableAsync<MeasurementResult>().Wait();
        }

        public Task<List<MeasurementResult>> GetMeasurementsAsync()
        {
            return _database.Table<MeasurementResult>().ToListAsync();
        }

        public Task<int> SaveMeasurementAsync(MeasurementResult measurement)
        {
            return _database.InsertAsync(measurement);
        }

        public Task<int> DeleteMeasurementAsync(MeasurementResult measurement)
        {
            return _database.DeleteAsync(measurement);
        }
    }
}
