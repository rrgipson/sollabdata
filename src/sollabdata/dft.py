"""DFT calculation output.

A Class for Compiling DFT Data parsed from cclib-generated json files.
Each json file holds a small ccframe-style table for one compound/calculation: one row per job stage
(e.g. opt, sp, freq) and one column per cclib attribute (scfenergies, atomcoords, vibfreqs, etc.).
Child class of :class:`~sollabdata.labdata.LabData`, which handles info_df
bookkeeping, copying, dropping, plotting, and writing.
Written by Claude, based on the parsing/analysis logic in DFT/CustomFxns.py.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from cclib.parser.utils import convertor

from .labdata import LabData

#Some global variables
ATOM_NAMES = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 26: 'Fe'}

#json is the raw format here, not csv, so read() overrides the base extension list
DATA_EXTENSIONS = ('.json',)


class DFT(LabData):
    '''DFT Data Class for cclib-parsed DFT calculation results with parent LabData.'''
    def __init__(self,
                 experiment_df: pd.DataFrame | None = None,
                 info_csv: str | None = None,
                 info_df: pd.DataFrame | None = None,
                 processing_metadata=None,
                 path_to_raw_data: str | None = None,
                 labels: dict | None = None,
                 **kwargs) -> None:
        '''See LabData.__init__ for the shared arguments; path_to_raw_data holds json files here.'''
        super().__init__(experiment_df, info_csv, info_df, processing_metadata, path_to_raw_data, labels, **kwargs)
        #drop empty columns -- check self.info_df, since process() may have just built it from info_csv
        if self.info_df is not None:
            self.info_df.drop(self.info_df.columns[self.info_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    def read(self, path_to_raw_data, info_csv):
        '''Given a folder with an info csv and cclib-generated data json files, read the files, and fill in the info df without data.'''
        #Resolve info_csv the same way the base class does: a bare name is taken
        #relative to the data folder, a real path is used as given
        info_path = None
        if info_csv is not None:
            info_path = info_csv if os.path.isfile(info_csv) else os.path.join(path_to_raw_data, info_csv)
        info_name = None if info_csv is None else os.path.basename(info_csv)
        #Collect the json files from the top level of the folder only
        data_files = sorted(entry.name for entry in os.scandir(path_to_raw_data)
                            if entry.is_file() and entry.name.lower().endswith(DATA_EXTENSIONS)
                            and entry.name != info_name)
        if info_path is not None:
            self.info_df = pd.read_csv(info_path)
        elif self.info_df is None:
            #Generate a generic info df based on files
            self.info_df = pd.DataFrame({'id': [os.path.splitext(f)[0] for f in data_files],
                                         'File': data_files})
        #Create a column in info_df to hold the data
        self.info_df['data'] = pd.Series(dtype='object')
        return data_files

    def load(self, path_to_raw_data, data_files, **kwargs):
        '''Read each cclib-generated json file into info_df['data'] as a table (rows = job stages, columns = cclib attributes).'''
        for i, row in self.info_df.iterrows():
            if self.info_df.at[i, 'File'] in data_files:
                self.info_df.at[i, 'data'] = pd.read_json(os.path.join(path_to_raw_data, row['File']), **kwargs)
        return True

    def _get_data(self, id) -> pd.DataFrame:
        '''Look up a compound's data table by id (or by integer info_df index).'''
        idx = id if isinstance(id, (int, np.integer)) else self.get_idx_from_id([id])[0]
        return self.info_df.at[idx, 'data']

    def get_param(self, id, param, file_ind=None):
        '''Pull a cclib attribute for a compound.
        id - the compound's id (or an integer index into info_df).
        file_ind - selects a specific job-stage row (e.g. 0=opt, 1=sp, 2=freq).
        If file_ind is not given, returns the scalar value if the attribute is the same across job stages,
        or the longest list value (e.g. the full optimization trajectory rather than a single-point job) otherwise.'''
        data = self._get_data(id)
        if file_ind is not None:
            return data.at[file_ind, param]
        by_length = {}
        for value in data[param]:
            if isinstance(value, list):
                by_length[len(value)] = value
            else:
                return value
        return by_length[max(by_length)] if by_length else None

    def get_atoms(self, id, df=None, file_ind=None) -> pd.DataFrame:
        '''Return a df of atom numbers and element symbols (initializes the dataframe if not given).'''
        atom_list = [ATOM_NAMES.get(num, 'ATOM NOT FOUND') for num in self.get_param(id, 'atomnos', file_ind)]
        if df is None:
            df = pd.DataFrame(data=atom_list, columns=['Atom'])
        else:
            df['Atom'] = atom_list
        df.insert(0, '#', range(1, len(atom_list)+1))
        return df

    def get_opt_geom(self, id, df, file_ind=None) -> pd.DataFrame:
        '''Add the final geometry's x,y,z coordinates to df.'''
        df_temp = pd.DataFrame(data=self.get_param(id, 'atomcoords', file_ind)[-1], columns=['x', 'y', 'z'])
        df[['x', 'y', 'z']] = df_temp[['x', 'y', 'z']]
        return df

    def msa(self, id, df, file_ind=None) -> pd.DataFrame:
        '''Add Mulliken charge and spin population columns to df.'''
        df['Charge'] = self.get_param(id, 'atomcharges', file_ind)['mulliken']
        df['Spin'] = self.get_param(id, 'atomspins', file_ind)['mulliken']
        df = df.round({'Charge': 3, 'Spin': 3})
        return df

    def get_geom(self, id, file_ind=None, mulliken=True) -> pd.DataFrame:
        '''Convenience wrapper: atom identities, final geometry, and (optionally) Mulliken charges/spins.'''
        df = self.get_atoms(id, file_ind=file_ind)
        df = self.get_opt_geom(id, df, file_ind=file_ind)
        if mulliken:
            df = self.msa(id, df, file_ind=file_ind)
        return df

    def get_vibs(self, id, file_ind=None) -> pd.DataFrame:
        '''Return a df of vibrational frequencies and IR intensities.'''
        vib_df = pd.DataFrame()
        vib_df['Freqs'] = self.get_param(id, 'vibfreqs', file_ind)
        vib_df['IR'] = self.get_param(id, 'vibirs', file_ind)
        return vib_df

    def get_thermo(self, id, freq_idx=2, reactant_id=None, reactant: DFT | None = None, react_freq_idx=2,
                   units: str = 'kcal/mol', skipscf=False) -> dict:
        '''Parse E, E+zpve, H, S, and G in the specified units for a compound (or the delta vs. a reactant if reactant_id is given).
        reactant defaults to self, so reactant_id can reference another row of the same DFT object.
        Returns a dict of the thermochemical values (and temperature/pressure).'''
        thermo_data = {}
        if not skipscf:
            thermo_data['E'] = convertor(self.get_param(id, 'scfenergies', freq_idx)[-1], 'eV', units)
            thermo_data['Ezpve'] = thermo_data['E'] + convertor(self.get_param(id, 'zpve', freq_idx), 'hartree', units)
        thermo_data['H'] = convertor(self.get_param(id, 'enthalpy', freq_idx), 'hartree', units)
        thermo_data['S'] = convertor(self.get_param(id, 'entropy', freq_idx), 'hartree', units)
        thermo_data['G'] = convertor(self.get_param(id, 'freeenergy', freq_idx), 'hartree', units)
        thermo_data['T_K'] = self.get_param(id, 'temperature', freq_idx)
        thermo_data['T_C'] = thermo_data['T_K'] - 273.15
        thermo_data['P_atm'] = self.get_param(id, 'pressure', freq_idx)
        thermo_data['units'] = units

        #Compare to reactant
        if reactant_id is not None:
            reactant = self if reactant is None else reactant
            if thermo_data['T_K'] != reactant.get_param(reactant_id, 'temperature', react_freq_idx):
                print('Warning: Different temperatures between reactant and product data.')
                thermo_data['T_K_reactant'] = reactant.get_param(reactant_id, 'temperature', react_freq_idx)
            if not skipscf:
                thermo_data['E'] -= convertor(reactant.get_param(reactant_id, 'scfenergies', react_freq_idx)[-1], 'eV', units)
                thermo_data['Ezpve'] -= (convertor(reactant.get_param(reactant_id, 'scfenergies', react_freq_idx)[-1], 'eV', units)
                                      + convertor(reactant.get_param(reactant_id, 'zpve', react_freq_idx), 'hartree', units))
            thermo_data['H'] -= convertor(reactant.get_param(reactant_id, 'enthalpy', react_freq_idx), 'hartree', units)
            thermo_data['S'] -= convertor(reactant.get_param(reactant_id, 'entropy', react_freq_idx), 'hartree', units)
            thermo_data['G'] -= convertor(reactant.get_param(reactant_id, 'freeenergy', react_freq_idx), 'hartree', units)

        for key in thermo_data:
            if key != 'units':
                thermo_data[key] = float(thermo_data[key])
        return thermo_data

    def get_scf(self, id, file_ind=None, reactant_id=None, reactant: DFT | None = None, react_file_ind=None,
                   units: str = 'kcal/mol') -> float:
        '''Parse E in the specified units for a compound (or the delta vs. a reactant if reactant_id is given).
        reactant defaults to self, so reactant_id can reference another row of the same DFT object.
        Returns the scf energy value.'''
        scf = convertor(self.get_param(id, 'scfenergies', file_ind)[-1], 'eV', units)

        #Compare to reactant
        if reactant_id is not None:
            reactant = self if reactant is None else reactant
            scf -= convertor(reactant.get_param(reactant_id, 'scfenergies', react_file_ind)[-1], 'eV', units)

        return scf

    def get_dist(self, id, a1, a2, a3=None, file_ind=None, verbose=False):
        '''Get the a1-a2 distance in Angstroms (atoms numbered starting at 1, as in get_atoms/get_geom).
        If a3 is given, returns the a1-a2-a3 angle in degrees *instead* of the distance
        (verbose=True prints both bond distances along the way).
        Builds the geometry fresh each call rather than requiring a pre-computed geometry df.'''
        df = self.get_geom(id, file_ind=file_ind, mulliken=False)
        #get a1-a2 distance
        df21 = pd.concat([df.loc[(df['#']==a2),['x','y','z']],df.loc[(df['#']==a1),['x','y','z']]],ignore_index=True)
        diff12 = df21.diff().iloc[1]
        dist12 = np.sqrt(np.sum(diff12**2))
        dist_txt = '{start}-{stop} distance = {dist:.3f} A'
        if verbose:
            print(dist_txt.format(start=df.at[a1-1,'Atom'],stop=df.at[a2-1,'Atom'],dist=dist12))
        if a3 is not None:
            #get a2-a3 distance
            df23 = pd.concat([df.loc[(df['#']==a2),['x','y','z']],df.loc[(df['#']==a3),['x','y','z']]],ignore_index=True)
            diff23 = df23.diff().iloc[1]
            dist23 = np.sqrt(np.sum(diff23**2))
            if verbose:
                print(dist_txt.format(start=df.at[a2-1,'Atom'],stop=df.at[a3-1,'Atom'],dist=dist23))
            #get a1-a2-a3 angle
            #calculate the normalized dot product
            dotprod = ((diff12).div(dist12)).dot(diff23.div(dist23))
            angle_txt = '{start}-{mid}-{stop} angle = {ang:.1f} degrees'
            #get angle between with arccosine
            ang = np.degrees(np.arccos(dotprod))
            if verbose:
                print(angle_txt.format(start=df.at[a1-1,'Atom'],mid=df.at[a2-1,'Atom'],stop=df.at[a3-1,'Atom'],ang=ang))
            return ang
        else:
            return dist12

    def get_dihedral(self, id, a1, a2, a3, a4, file_ind=None, verbose=False):
        '''Calculate the a1-a2-a3-a4 dihedral angle in degrees (atoms numbered starting at 1, as in get_atoms/get_geom).
        Builds the geometry fresh each call rather than requiring a pre-computed geometry df.'''
        df = self.get_geom(id, file_ind=file_ind, mulliken=False)
        #Get coordinates for all four atoms
        p1 = df.loc[df['#'] == a1, ['x', 'y', 'z']].values[0]
        p2 = df.loc[df['#'] == a2, ['x', 'y', 'z']].values[0]
        p3 = df.loc[df['#'] == a3, ['x', 'y', 'z']].values[0]
        p4 = df.loc[df['#'] == a4, ['x', 'y', 'z']].values[0]

        #Calculate vectors between consecutive atoms
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        #Normalize b2
        b2_norm = b2 / np.linalg.norm(b2)

        #Calculate normal vectors to the two planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        #Normalize the normal vectors
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        #Calculate the dihedral angle
        m1 = np.cross(n1, b2_norm)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)

        dihedral = np.degrees(np.arctan2(y, x))

        if verbose:
            atom1 = df.loc[df['#'] == a1, 'Atom'].values[0]
            atom2 = df.loc[df['#'] == a2, 'Atom'].values[0]
            atom3 = df.loc[df['#'] == a3, 'Atom'].values[0]
            atom4 = df.loc[df['#'] == a4, 'Atom'].values[0]

            dihedral_txt = '{a1}-{a2}-{a3}-{a4} dihedral = {angle:.2f}°'
            print(dihedral_txt.format(a1=atom1, a2=atom2, a3=atom3, a4=atom4, angle=dihedral))

        return dihedral


#: Backwards-compatible alias for the old class name used in existing notebooks.
DFT_Data = DFT


# ---------------------------------------------------------------------------
# Code written by RG (Robert Gipson).
# Claude (Opus 5) reviewed and adjusted this file:
#   - Added the missing `os` / `numpy` / `pandas` imports; the module used os.walk,
#     pd and np without importing any of them, so it could not be imported.
#   - Replaced the string parameter annotations on __init__ with real type hints,
#     and the stale 'DFT_Data' annotations with the actual DFT class (alias kept).
#   - The "drop unnamed columns" step now checks `self.info_df`, so it also runs
#     when info_df came from info_csv via process().
#   - read(): resolves info_csv relative to path_to_raw_data when it is a bare file
#     name (it was previously read relative to the cwd while the json files were
#     read relative to the data folder), compares against the info file's basename,
#     and uses os.scandir instead of walking a tree it then discarded.
#   - load(): os.path.join instead of string concatenation, so path_to_raw_data no
#     longer needs a trailing separator; accepts **kwargs from process().
#   - get_dist(): docstring corrected -- with a3 it returns only the angle, not the
#     distances (the code was right, the docstring was not).
# ---------------------------------------------------------------------------
