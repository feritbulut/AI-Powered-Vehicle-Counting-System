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

        // Creates database connection and table
        public MeasurementDatabase(string dbPath)
        {
            _database = new SQLiteAsyncConnection(dbPath);
            _database.CreateTableAsync<MeasurementResult>().Wait();
        }

        // Gets all measurements from database
        public Task<List<MeasurementResult>> GetMeasurementsAsync()
        {
            return _database.Table<MeasurementResult>().ToListAsync();
        }

        // Saves a new measurement
        public Task<int> SaveMeasurementAsync(MeasurementResult measurement)
        {
            return _database.InsertAsync(measurement);
        }

        // Deletes a measurement
        public Task<int> DeleteMeasurementAsync(MeasurementResult measurement)
        {
            return _database.DeleteAsync(measurement);
        }
    }
}
