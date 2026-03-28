import unittest

from car_data_etl import DEFAULT_VEHICLE_YEARS, normalize_vehicle_years


class NormalizeVehicleYearsTests(unittest.TestCase):
    def test_default_years_are_returned_as_a_new_list(self) -> None:
        normalized = normalize_vehicle_years()
        normalized.append(2020)

        self.assertEqual(list(DEFAULT_VEHICLE_YEARS), [2017])
        self.assertEqual(normalize_vehicle_years(), [2017])

    def test_empty_years_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            normalize_vehicle_years([])


if __name__ == "__main__":
    unittest.main()
