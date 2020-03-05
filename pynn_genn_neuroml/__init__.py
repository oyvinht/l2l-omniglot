from itertools import repeat
from pyNN.core import ezip
import math
# https://libneuroml.readthedocs.io/en/latest/
import neuroml
import neuroml.writers as writers
import numpy
from pyNN import common
import pyNN.common.projections as pynn_projections
import pyNN.connectors as pynn_connectors
import pyNN.standardmodels.cells as pynn_cells
import inspect
import logging

log = logging.getLogger('PyNNGeNNNeuroML')
log.setLevel(logging.INFO)

layer_size_limit = 512

def export(filename, populations, projections):
    nml_doc = neuroml.NeuroMLDocument(id='PyNNGeNNNeuroML')
    nml_net = neuroml.Network(id='network')
    nml_doc.networks.append(nml_net)
    _add_populations(nml_doc, nml_net, populations)
    _add_projections(nml_doc, nml_net, projections)
    _write_nml_doc(nml_doc, filename)


import pprint

def _add_populations(nml_doc, nml_net, populations):
    for pop_idx, pop in enumerate(populations):
        pop_label =  pop.label.replace(' ', '_')
        nml_celltype = _find_matching_nml_celltype(pop.celltype)
        pynn_cell = _make_matching_pynn_cell(pop.celltype)
        nml_cell = nml_celltype(id='{}_{}'.format(nml_celltype.__name__, pop_label))
        
        # Add params on a best effort basis
        members = _get_member_data_items(nml_cell)
        for member in members:
            param = member.get_name()
            if pynn_cell.has_parameter(param):
                value_generator = pynn_cell.parameter_space[param].base_value
                nml_cell.__setattr__(param, float(value_generator))
            else:
                value = float(0.0) if member.get_data_type() == 'xs:float' else None
                nml_cell.__setattr__(param, value)
        nml_doc.append(nml_cell)

        #if pop.celltype.__class__.__name__ == 'SpikeSourceArray':
            #print('pop size: {} (of type {}) in population "{}"'.format(pop.size, pop.celltype, pop.label))
            #for cell_idx, cell in enumerate(pop.all()):
                #print(cell.get_parameters()['spike_times'])

            #for cell in pop.__iter__():
                #print(cell)
                #print(type(cell))
                #print(dir(cell))
        
        # Add instances to net
        nml_pop = neuroml.Population(id=pop_label, size=min(pop.size,layer_size_limit), type='populationList', component=nml_cell.id)
        for cell_idx,cell in enumerate(pop.all()):
            if cell_idx < layer_size_limit:

                # Put cells in squares as far as possible
                side_len = math.floor(math.sqrt(min(len(pop), layer_size_limit)))
                location = neuroml.Location(x=cell_idx % side_len,
                                            y=math.floor(cell_idx / side_len),
                                            z=pop_idx * 4)
                instance = neuroml.Instance(id=pop.id_to_index(cell), location=location)
                nml_pop.instances.append(instance)
        nml_net.populations.append(nml_pop)

def _add_projections(nml_doc, nml_net, projections):
    for proj_idx, proj in enumerate(projections):
        import pprint
        
        nml_pre_celltype = _find_matching_nml_celltype(proj.pre.celltype)
        nml_post_celltype = _find_matching_nml_celltype(proj.post.celltype)
        pre_pop_name = proj.pre.label.replace(' ', '_')
        post_pop_name = proj.post.label.replace(' ', '_')
        pynn_cell = _make_matching_pynn_cell(proj.post.celltype)
        syn = None

        syn_id = 'syn_{}'.format(proj_idx)

        if proj.receptor_type == 'inhibitory':
            tau_key = 'tau_syn_I',
            erev_key = 'e_rev_I'
        else:
            tau_key = 'tau_syn_E'
            erev_key = 'e_rev_I'
        if 'cond_exp' in nml_post_celltype.__name__:
            syn = neuroml.ExpCondSynapse(id=syn_id)
            syn.__setattr__('e_rev', proj.post.celltype.parameter_space[erev_key].base_value)
            nml_doc.exp_cond_synapses.append(syn)
        elif 'curr_exp' in nml_post_celltype.__name__:
            syn = neuroml.ExpCurrSynapse(id=syn_id)
            nml_doc.exp_curr_synapses.append(syn)

        #print(pynn_cell.parameter_space[tau_key])
        syn.tau_syn = 5.0 # TODO: Pick base value for param
        
        nml_proj = neuroml.Projection(id='projection_{}'.format(proj_idx),
                                      presynaptic_population=pre_pop_name,
                                      postsynaptic_population=post_pop_name,
                                      synapse=syn.id)
        nml_net.projections.append(nml_proj)
        # Add connections
        if isinstance(proj._connector, pynn_connectors.AllToAllConnector):
            for pre_cell_idx, pre_cell in enumerate(proj.pre.all()):
                if pre_cell_idx < layer_size_limit:
                    for post_cell_idx, post_cell in enumerate(proj.post.all()):
                        if post_cell_idx < layer_size_limit:
                            # TODO: Add syn params
                            nml_conn = _make_connection(proj_idx,
                                                        pre_pop_name,
                                                        pre_cell_idx,
                                                        nml_pre_celltype.__name__,
                                                        post_pop_name,
                                                        post_cell_idx,
                                                        nml_post_celltype.__name__)
                            nml_proj.connections.append(nml_conn)
        elif isinstance(proj._connector, pynn_connectors.FromListConnector):
            # TODO: Columns could be configured to be something else than weight, delay
            for pre_idx, post_idx, weight, delay in proj._connector.conn_list.tolist():
                #print('pre: {}, post: {}, w:{}, d:{}'.format(pre_idx, post_idx, weight, delay))
                if pre_idx < layer_size_limit and post_idx < layer_size_limit:
                    nml_conn = _make_connection(proj_idx,
                                                pre_pop_name,
                                                int(pre_idx),
                                                nml_pre_celltype.__name__,
                                                post_pop_name,
                                                int(post_idx),
                                                nml_post_celltype.__name__)
                    nml_proj.connections.append(nml_conn)


def _make_connection(proj_idx, pre_pop_name, pre_cell_idx, pre_celltype_name, post_pop_name, post_cell_idx, post_celltype_name):
    return neuroml.Connection(id='{}'.format(proj_idx),
                              pre_cell_id='../{}/{}/{}_{}'.format(pre_pop_name,
                                                                  pre_cell_idx,
                                                                  pre_celltype_name,
                                                                  pre_pop_name),
                              post_cell_id='../{}/{}/{}_{}'.format(post_pop_name,
                                                                   post_cell_idx,
                                                                   post_celltype_name,
                                                                   post_pop_name))
    
def _write_nml_doc(nml_doc, filename):
    writers.NeuroMLWriter.write(nml_doc, filename)
    log.info('Wrote NeuroML document to "{}"'.format(filename))
    from neuroml.utils import validate_neuroml2
    validate_neuroml2(filename)

def _find_matching_nml_celltype(celltype):
    """Find a neuroml cell type that best matches the given pyNN(GeNN) type."""
    if isinstance(celltype, pynn_cells.IF_curr_exp):
        return neuroml.IF_curr_exp
    elif isinstance(celltype, pynn_cells.IF_cond_exp):
        return neuroml.IF_cond_exp
    elif isinstance(celltype, pynn_cells.SpikeSourceArray):
        return neuroml.SpikeArray
    else:
        # TODO: Add relevant types
        log.error('Could not find libneuroml match for celltype "{}"'.format(celltype))
        return celltype

def _get_member_data_items(nml_cell):
    """Traverse neuroml cell class hierarchy to collect cell parameters."""
    if  isinstance(nml_cell, neuroml.SpikeArray) or isinstance(nml_cell, neuroml.BaseCell) or issubclass(nml_cell, neuroml.BaseCell):
        return _get_member_data_items(nml_cell.superclass) + nml_cell.member_data_items_
    else:
        return []

def _make_matching_pynn_cell(celltype):
    if isinstance(celltype, pynn_cells.IF_curr_exp):
        return pynn_cells.IF_curr_exp()
    elif isinstance(celltype, pynn_cells.IF_cond_exp):
        return pynn_cells.IF_cond_exp()
    elif isinstance(celltype, pynn_cells.SpikeSourceArray):
        return pynn_cells.SpikeSourceArray()
    else:
        # TODO: Add relevant types
        print('not found {}'.format(celltype))
        log.error('Could not find PyNN match for celltype "{}"'.format(celltype))
        return celltype
        
