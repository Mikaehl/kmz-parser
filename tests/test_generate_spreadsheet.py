import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import generate_spreadsheet


KML_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Folder><name>Sites</name>
    <Placemark><name>PAR-001</name><styleUrl>#siteIcon</styleUrl><Point><coordinates>2.35,48.86,0</coordinates></Point></Placemark>
  </Folder>
  <Folder><name>Manholes</name>
    <Placemark><name>PAR-001-MH1</name><styleUrl>#squareIcon</styleUrl><Point><coordinates>2.36,48.87,0</coordinates></Point></Placemark>
  </Folder>
  <Placemark><name>Site outside folder</name><styleUrl>#siteIcon</styleUrl><Point><coordinates>2.37,48.88,0</coordinates></Point></Placemark>
</Document></kml>"""

JOINTS_KML_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
    <Folder><name>Manholes</name>
        <Placemark><name>PAR-001-MH1</name><Point><coordinates>2.35,48.86,0</coordinates></Point></Placemark>
        <Placemark><name>001-001</name><Point><coordinates>2.36,48.87,0</coordinates></Point></Placemark>
        <Placemark><name>001-002</name><Point><coordinates>2.37,48.88,0</coordinates></Point></Placemark>
    </Folder>
    <Folder><name>Cables</name>
        <Placemark id="CABLE-001"><name>PAR-001_MH1_001-001</name><LineString><coordinates>2.35,48.86,0 2.36,48.87,0</coordinates></LineString></Placemark>
        <Placemark><name>001-001_001-002</name><LineString><coordinates>2.36,48.87,0 2.37,48.88,0</coordinates></LineString></Placemark>
        <Placemark><name>001-001_001-002</name><LineString><coordinates>2.36,48.87,0 2.37,48.88,0</coordinates></LineString></Placemark>
        <Placemark id="AUX-001"><name>Auxiliary cable</name><LineString><coordinates>2.38,48.89,0 2.37,48.88,0</coordinates></LineString></Placemark>
    </Folder>
</Document></kml>"""


class SpreadsheetGeneratorTests(unittest.TestCase):
    def _write_kmz(self, path: Path) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("doc.kml", KML_CONTENT)

    def test_generates_sites_sheet_from_sites_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            kmz_path = root / "network.kmz"
            output_path = root / "network.xlsx"
            self._write_kmz(kmz_path)

            with patch(
                "generate_spreadsheet.get_address_and_company",
                return_value=("10 rue Test, Paris", "Entreprise Test"),
            ) as address_lookup:
                result_path = generate_spreadsheet.generate_spreadsheet(
                    [kmz_path], output_path
                )

            sites = pd.read_excel(result_path, sheet_name="Sites")
            manholes = pd.read_excel(result_path, sheet_name="Manholes")

        self.assertEqual(len(sites), 1)
        self.assertEqual(sites.loc[0, "id_site"], "PAR-001")
        self.assertAlmostEqual(sites.loc[0, "latitude"], 48.86)
        self.assertAlmostEqual(sites.loc[0, "longitude"], 2.35)
        self.assertEqual(sites.loc[0, "adresse"], "10 rue Test, Paris")
        self.assertEqual(sites.loc[0, "entreprise"], "Entreprise Test")
        self.assertEqual(sites.loc[0, "fichier_kmz"], "network.kmz")
        self.assertEqual(list(manholes["nom_point"]), ["PAR-001-MH1", "Site outside folder"])
        address_lookup.assert_called_once_with(48.86, 2.35, generate_spreadsheet.REQUEST_TIMEOUT_SECONDS)

    @patch("generate_spreadsheet.requests.get")
    def test_looks_up_companies_at_the_reverse_geocoded_address(self, mock_get: Mock) -> None:
        address_response = Mock()
        address_response.json.return_value = {
            "features": [
                {
                    "properties": {
                        "label": "12 rue de l'École, 75001 Paris",
                        "housenumber": "12",
                        "street": "Rue de l'École",
                        "postcode": "75001",
                        "city": "Paris",
                    }
                }
            ]
        }
        enterprise_response = Mock()
        enterprise_response.json.return_value = {
            "results": [
                {
                    "nom_complet": "Entreprise à cette adresse",
                    "matching_etablissements": [
                        {"adresse": "12 RUE DE L ECOLE 75001 PARIS", "etat_administratif": "A"}
                    ],
                },
                {
                    "nom_complet": "Entreprise au numéro voisin",
                    "matching_etablissements": [
                        {"adresse": "14 RUE DE L ECOLE 75001 PARIS", "etat_administratif": "A"}
                    ],
                },
            ]
        }
        mock_get.side_effect = [address_response, enterprise_response]

        address, company = generate_spreadsheet.get_address_and_company(48.86, 2.35)

        self.assertEqual(address, "12 rue de l'École, 75001 Paris")
        self.assertEqual(company, "Entreprise à cette adresse")
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_get.call_args.args[0], generate_spreadsheet.ENTERPRISE_API_URL)
        self.assertEqual(mock_get.call_args.kwargs["params"]["q"], "12 Rue de l'École 75001 Paris")

    def test_joints_sheet_has_a_column_per_cable_with_its_id_at_both_ends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            kmz_path = root / "joints.kmz"
            output_path = root / "joints.xlsx"
            with zipfile.ZipFile(kmz_path, "w") as archive:
                archive.writestr("doc.kml", JOINTS_KML_CONTENT)

            generate_spreadsheet.generate_spreadsheet([kmz_path], output_path)
            joints = pd.read_excel(output_path, sheet_name="Joints").fillna("")
            cables = pd.read_excel(output_path, sheet_name="Cables").fillna("")

        self.assertEqual(
            list(cables["id_cable"]),
            ["CABLE-001", "001-001_001-002", "001-001_001-002#2", "AUX-001"],
        )
        self.assertEqual(
            list(joints.columns),
            [
                "joint_id",
                "manhole",
                "CABLE-001",
                "001-001_001-002",
                "001-001_001-002#2",
                "AUX-001",
            ],
        )
        indexed = joints.set_index("manhole")
        self.assertEqual(indexed.loc["PAR-001-MH1", "CABLE-001"], "CABLE-001")
        self.assertEqual(indexed.loc["001-001", "CABLE-001"], "CABLE-001")
        self.assertEqual(indexed.loc["001-001", "001-001_001-002"], "001-001_001-002")
        self.assertEqual(indexed.loc["001-002", "001-001_001-002#2"], "001-001_001-002#2")
        self.assertEqual(indexed.loc["001-002", "AUX-001"], "AUX-001")
        self.assertEqual(indexed.loc["PAR-001-MH1", "001-001_001-002"], "")


if __name__ == "__main__":
    unittest.main()