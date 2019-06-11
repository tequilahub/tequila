import unittest
# from .context import openvqe
import openvqe.parameters as param



class TestParameters(unittest.TestCase):

    def create_filename(self, name):
        filename = type(self).__name__ + "_" + str(name) + ".out"
        self.filenames.append(filename)
        return filename

    def setUp(self) -> None:
        # store filenames and clean up later
        self.filenames = []

    def tearDown(self) -> None:
        import os
        for filename in self.filenames: os.remove(filename)

    def init_parameters(self, c):
        p = c()
        for i, key in enumerate(p.__dict__.keys()):
            try:
                if isinstance(p.__dict__[key], param.ParametersBase):
                    p.__dict__[key] = self.init_parameters(c=type(c.__dict__[key]))
                else: p.__dict__[key] = type(c.__dict__[key])(i)
            except KeyError:
                if isinstance(p.__dict__[key], param.ParametersBase):
                    p.__dict__[key] = self.init_parameters(c=type(p.__dict__[key]))
                else: p.__dict__[key] = type(p.__dict__[key])(i)
        return p

    def test_io(self):

        all_parameters = [
            param.ParametersBase,
            param.ParametersQC,
            param.ParametersPsi4,
            param.ParametersAnsatz,
            param.ParametersUCC,
            param.ParametersHamiltonian
        ]

        for c in all_parameters:
            p = self.init_parameters(c)
            filename = self.create_filename(p.name())
            p.print_to_file(filename=filename)
            p2 = c.read_from_file(filename=filename)
            self.assertEqual(p, p2)

    def test_qcio(self):
        ParametersQC = param.ParametersQC
        geomfile = self.create_filename("geometry_test")
        test_string = "3\ncomment\nh 1.0 2.0 3.0\nh 4.0 5.0 6.0\no 7.0 8.0 9.0\n"
        with open(geomfile, 'a+') as f:
            f.write(test_string)

        read_string, comment = ParametersQC.read_xyz_from_file(geomfile)
        self.assertEqual("3\ncomment\n" + read_string, test_string)
        self.assertEqual(comment.strip(), "comment")

        data1 = ParametersQC.convert_to_list(read_string)
        param0 = ParametersQC()
        param0.geometry = read_string
        data2 = param0.get_geometry()
        self.assertEqual(data1, data2)

        paramfile = self.create_filename("extra_test_for_qc_parameters")
        param1 = ParametersQC()
        param1.description = "change comment for testing"
        param1.print_to_file(filename=paramfile)

        param1x = ParametersQC.read_from_file(filename=paramfile)

        self.assertEqual(param1, param1x)


if __name__ == '__main__':
    unittest.main()
