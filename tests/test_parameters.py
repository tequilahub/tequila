import unittest
# from .context import openvqe
from openvqe.parameters import Parameters


class TestParameters(unittest.TestCase):
    def setUp(self) -> None:
        self.filenames = {"parameters": type(self).__name__ + "_parameters.out",
                          "geometry": type(self).__name__ + "_geometry.out"}
        import os
        for key in self.filenames:
            if os.path.exists(self.filenames[key]):
                for i in range(100):
                    self.filenames[key] += "(" + str(i) + ")"
                    if os.path.exists(self.filenames[key]):
                        continue
                    else:
                        break
                if os.path.exists(self.filenames[key]):
                    raise Exception(
                        "the file " + self.filenames[
                            key] + " does already exist. Make sure you don't need it and delete it in order for the test to run")

    def tearDown(self) -> None:
        import os
        for filename in self.filenames.values(): os.remove(filename)

    filenames: dict

    def test_io(self):

        geomfile = self.filenames['geometry']
        test_string = "3\ncomment\nh 1.0 2.0 3.0\nh 4.0 5.0 6.0\no 7.0 8.0 9.0\n"
        with open(geomfile, 'a+') as f:
            f.write(test_string)

        read_string = Parameters.QC.read_xyz_from_file(geomfile)
        self.assertEqual("3\ncomment\n" + read_string, test_string)

        data1 = Parameters.QC.convert_to_list(read_string)
        param = Parameters()
        param.qc.geometry = read_string
        data2 = param.qc.get_geometry()
        self.assertEqual(data1, data2)

        paramfile = self.filenames['parameters']
        param1 = Parameters()
        name1 = "name1"
        param1.comment = "change comment for testing"
        param1.print_to_file(filename=paramfile, name=name1)

        param2 = Parameters()
        name2 = "name2"
        param2.comment = "change comment for testing too"
        param2.print_to_file(filename=paramfile, name=name2)

        param1x = Parameters.read_from_file(filename=paramfile, name=name1)
        param2x = Parameters.read_from_file(filename=paramfile, name=name2)

        self.assertEqual(param1, param1x)
        self.assertEqual(param2, param2x)
        self.assertFalse(param1 == param2x)
        self.assertFalse(param2 == param1x)


if __name__ == '__main__':
    unittest.main()
