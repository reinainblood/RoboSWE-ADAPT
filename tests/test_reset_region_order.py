from types import SimpleNamespace

import pytest

from pairing.reset_region_order import CounterRegionOrder, anchored_regions, anchors_from_metadata


class Counter:
    name='counter'
    reverse=False
    def get_reset_regions(self,env,ref=None,loc='nn'):
        values=[{'size':[1.,.5],'offset':[-1.,0.,.4]}, {'size':[2.,.5],'offset':[1.,0.,.4]}]
        if self.reverse:values.reverse()
        return {f'geom_{i}':v for i,v in enumerate(values)}


def test_recorded_order_restores_geometry_names_without_extra_calls():
    c=Counter();original=Counter.get_reset_regions
    with CounterRegionOrder() as capture:
        capture.install(Counter);a=c.get_reset_regions(None,ref=SimpleNamespace(name='sink'),loc='left_right')
    assert Counter.get_reset_regions is original
    c.reverse=True
    with CounterRegionOrder(capture.payload()) as replay:
        replay.install(Counter);b=c.get_reset_regions(None,ref=SimpleNamespace(name='sink'),loc='left_right')
    assert a==b
    assert list(a)==list(b)
    assert replay.payload()==capture.payload()
    assert Counter.get_reset_regions is original


def test_query_mismatch_restores_upstream_method():
    c=Counter();original=Counter.get_reset_regions
    with CounterRegionOrder() as capture:
        capture.install(Counter);c.get_reset_regions(None,loc='left')
    with pytest.raises(ValueError,match='query sequence'):
        with CounterRegionOrder(capture.payload()) as replay:
            replay.install(Counter);c.get_reset_regions(None,loc='right')
    assert Counter.get_reset_regions is original


def test_geometry_change_is_not_silently_accepted():
    c=Counter()
    with CounterRegionOrder() as capture:
        capture.install(Counter);c.get_reset_regions(None)
    reference=capture.payload();reference['calls'][0]['regions'][0][1]['size']=[9.,9.]
    with pytest.raises(ValueError,match='geometry'):
        with CounterRegionOrder(reference) as replay:
            replay.install(Counter);c.get_reset_regions(None)


def test_frozen_metadata_can_anchor_the_original_region_assignment():
    metadata={'object_cfgs':[{'placement':{'fixture':'counter','sample_region_kwargs':{'ref':'sink','loc':'left_right','min_size':[.1,.1]}},
        'reset_region':{'name':'geom_1','size':[1.,.5],'offset':[-1.,0.,.4]}}]}
    anchors=anchors_from_metadata(metadata)
    actual=Counter().get_reset_regions(None)
    result=anchored_regions(actual,{'fixture':'counter','ref':'sink','loc':'left_right'},anchors)
    assert result['geom_1']['offset'][0]==-1
    assert list(result)==['geom_0','geom_1']
    assert actual['geom_0']['offset'][0]==-1


def test_modal_roundoff_matching_preserves_native_geometry():
    from pairing.reset_region_order import reorder_regions
    native={'size':[1.0,.65],'offset':[.5,0.,.46]}
    expected=[['geom_0',{'size':[.9999999999999998,.65],'offset':[.5000000000000001,0.,.46]}]]
    with pytest.raises(ValueError):reorder_regions({'renamed':native},expected)
    result=reorder_regions({'renamed':native},expected,atol=1e-12)
    assert result['geom_0'] is native
    bad=[['geom_0',{'size':[1.000001,.65],'offset':[.5,0.,.46]}]]
    with pytest.raises(ValueError):reorder_regions({'renamed':native},bad,atol=1e-12)
    with pytest.raises(ValueError):reorder_regions({'renamed':native},expected,atol=1e-6)


def test_roundoff_match_requires_unique_candidate():
    from pairing.reset_region_order import reorder_regions
    actual={'a':{'size':[1.+1e-13]},'b':{'size':[1.+3e-13]}}
    expected=[['geom_0',{'size':[1.+2e-13]}],['geom_1',{'size':[1.+4e-13]}]]
    with pytest.raises(ValueError,match='Ambiguous'):reorder_regions(actual,expected,atol=1e-12)
