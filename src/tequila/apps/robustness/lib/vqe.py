def make_ansatz(molecule, name):
    name = name.lower()

    if name == 'upccgsd':
        return molecule.make_upccgsd_ansatz(name='upccgsd')

    if 'spa' in name:
        ansatz = molecule.make_upccgsd_ansatz(name='SPA')
        if 'spa-gas' in name:
            ansatz += molecule.make_upccgsd_ansatz(name='GAS', include_reference=False)

        if 'spa-gs' in name:
            ansatz += molecule.make_upccgsd_ansatz(name='GS', include_reference=False)

        if 'spa-s' in name:
            ansatz += molecule.make_upccgsd_ansatz(name='S', include_reference=False)

        return ansatz

    raise NotImplementedError(f'Ansatz {name} not known')


def init_vqe_from_file():
    pass
