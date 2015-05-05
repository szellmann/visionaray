// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <iostream>
#include <memory>
#include <ostream>

#include <boost/algorithm/string.hpp>

#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>
#include <cover/VRViewer.h>

#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>

#include "drawable.h"
#include "state.h"
#include "visionaray_plugin.h"

namespace visionaray { namespace cover {


//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct Visionaray::impl : vrui::coMenuListener
{

    using check_box     = std::unique_ptr<vrui::coCheckboxMenuItem>;
    using menu          = std::unique_ptr<vrui::coMenu>;
    using radio_button  = std::unique_ptr<vrui::coCheckboxMenuItem>;
    using radio_group   = std::unique_ptr<vrui::coCheckboxGroup>;
    using sub_menu      = std::unique_ptr<vrui::coSubMenuItem>;

    impl()
        : drawable_ptr(new drawable)
    {
        init_state_from_config();
        drawable_ptr->update_state(state, dev_state);
    }

    osg::Node::NodeMask             objroot_node_mask;
    osg::ref_ptr<osg::Geode>        geode;
    osg::ref_ptr<drawable>          drawable_ptr;

    struct
    {
        menu                        main_menu;
        menu                        algo_menu;
        menu                        dev_menu;
        sub_menu                    main_menu_entry;
        sub_menu                    algo_menu_entry;
        sub_menu                    dev_menu_entry;

        // main menu
        check_box                   toggle_update_mode;

        // algo menu
        radio_group                 algo_group;
        radio_button                simple_button;
        radio_button                whitted_button;
        radio_button                pathtracing_button;

        // dev menu
        check_box                   toggle_bvh_display;
        check_box                   toggle_normal_display;
        check_box                   toggle_tex_coord_display;
    } ui;

    std::shared_ptr<render_state>   state;
    std::shared_ptr<debug_state>    dev_state;

    // init

    void init_state_from_config();
    void init_ui();

    // menu listener interface

    void menuEvent(vrui::coMenuItem* item);


    // control state

    void set_data_variance(data_variance var);
    void set_algorithm(algorithm algo);
    void set_show_bvh(bool show_bvh);
    void set_show_normals(bool show_normals);
    void set_show_tex_coords(bool show_tex_coords);
};

//-------------------------------------------------------------------------------------------------
// Read state from COVISE config
//

void Visionaray::impl::init_state_from_config()
{

    //
    //
    // <Visionaray>
    //     <DataVariance value="static"  />      <!-- "static" | "dynamic" -->
    //     <Algorithm    value="simple"  />      <!-- "simple" | "whitted" -->
    //     <CPUScheduler numThreads="16" />      <!-- numThreads:Integer   -->
    // </Visioaray>
    //
    //


    state     = std::make_shared<render_state>();
    dev_state = std::make_shared<debug_state>();


    // Read config

    using boost::algorithm::to_lower;

    auto algo_str       = covise::coCoviseConfig::getEntry("COVER.Plugin.Visionaray.Algorithm");
    auto data_var_str   = covise::coCoviseConfig::getEntry("COVER.Plugin.Visionaray.DataVariance");
    auto num_threads    = covise::coCoviseConfig::getInt("numThreads", "COVER.Plugin.Visionaray.CPUScheduler", 0);

    to_lower(algo_str);
    to_lower(data_var_str);


    // Update state

    if (algo_str == "whitted")
    {
        state->algo = Whitted;
    }
    else if (algo_str == "pathtracing")
    {
        state->algo = Pathtracing;
    }
    else
    {
        state->algo = Simple;
    }

    state->data_var     = data_var_str == "dynamic" ? Dynamic : Static;
    state->num_threads  = num_threads;
}

void Visionaray::impl::init_ui()
{
    using namespace vrui;

    ui.main_menu_entry.reset(new coSubMenuItem("Visionaray..."));
    opencover::cover->getMenu()->add(ui.main_menu_entry.get());

    // main menu

    ui.main_menu.reset(new coRowMenu("Visionaray", opencover::cover->getMenu()));
    ui.main_menu_entry->setMenu(ui.main_menu.get());


    ui.toggle_update_mode.reset(new coCheckboxMenuItem("Update scene per frame", state->data_var == Dynamic));
    ui.toggle_update_mode->setMenuListener(this);
    ui.main_menu->add(ui.toggle_update_mode.get());


    // algorithm submenu

    ui.algo_menu_entry.reset(new coSubMenuItem("Rendering algorithm..."));
    ui.main_menu->add(ui.algo_menu_entry.get());

    ui.algo_menu.reset(new coRowMenu("Rendering algorithm", ui.main_menu.get()));
    ui.algo_menu_entry->setMenu(ui.algo_menu.get());


    ui.algo_group.reset(new coCheckboxGroup( /* allow empty selection: */ false ));

    ui.simple_button.reset(new coCheckboxMenuItem("Simple", state->algo == Simple, ui.algo_group.get()));
    ui.simple_button->setMenuListener(this);
    ui.algo_menu->add(ui.simple_button.get());

    ui.whitted_button.reset(new coCheckboxMenuItem("Whitted", state->algo == Whitted, ui.algo_group.get()));
    ui.whitted_button->setMenuListener(this);
    ui.algo_menu->add(ui.whitted_button.get());

    ui.pathtracing_button.reset(new coCheckboxMenuItem("Pathtracing", state->algo == Pathtracing, ui.algo_group.get()));
    ui.pathtracing_button->setMenuListener(this);
    ui.algo_menu->add(ui.pathtracing_button.get());


    // dev submenu at the bottom!

    if (dev_state->debug_mode)
    {
        ui.dev_menu_entry.reset(new coSubMenuItem("Developer..."));
        ui.main_menu->add(ui.dev_menu_entry.get());

        ui.dev_menu.reset(new coRowMenu("Developer", ui.main_menu.get()));
        ui.dev_menu_entry->setMenu(ui.dev_menu.get());


        ui.toggle_bvh_display.reset(new coCheckboxMenuItem("Show BVH outlines", false));
        ui.toggle_bvh_display->setMenuListener(this);
        ui.dev_menu->add(ui.toggle_bvh_display.get());

        ui.toggle_normal_display.reset(new coCheckboxMenuItem("Show surface normals", false));
        ui.toggle_normal_display->setMenuListener(this);
        ui.dev_menu->add(ui.toggle_normal_display.get());

        ui.toggle_tex_coord_display.reset(new coCheckboxMenuItem("Show texture coordinates", false));
        ui.toggle_tex_coord_display->setMenuListener(this);
        ui.dev_menu->add(ui.toggle_tex_coord_display.get());
    }
}

void Visionaray::impl::menuEvent(vrui::coMenuItem* item)
{
    // main menu
    if (item == ui.toggle_update_mode.get())
    {
        set_data_variance(ui.toggle_update_mode->getState() ? Dynamic : Static);
    }

    // algorithm submenu
    if (item == ui.simple_button.get())
    {
        set_algorithm(Simple);
    }
    else if (item == ui.whitted_button.get())
    {
        set_algorithm(Whitted);
    }
    else if (item == ui.pathtracing_button.get())
    {
        set_algorithm(Pathtracing);
    }

    // dev submenu
    if (item == ui.toggle_bvh_display.get())
    {
        set_show_bvh(ui.toggle_bvh_display->getState());
    }

    if (item == ui.toggle_normal_display.get())
    {
        set_show_normals(ui.toggle_normal_display->getState());
    }

    if (item == ui.toggle_tex_coord_display.get())
    {
        set_show_tex_coords(ui.toggle_tex_coord_display->getState());
    }
}


//-------------------------------------------------------------------------------------------------
// Control state
//

void Visionaray::impl::set_data_variance(data_variance var)
{
    state->data_var = var;
    ui.toggle_update_mode->setState( var == Dynamic, false );
}

void Visionaray::impl::set_algorithm(algorithm algo)
{
    state->algo = algo;
    ui.simple_button->setState( algo == Simple, false );
    ui.whitted_button->setState( algo == Whitted, false );
    ui.pathtracing_button->setState( algo == Pathtracing, false );
}

void Visionaray::impl::set_show_bvh(bool show_bvh)
{
    dev_state->show_bvh = show_bvh;
    ui.toggle_bvh_display->setState( show_bvh, false );
}

void Visionaray::impl::set_show_normals(bool show_normals)
{
    dev_state->show_normals = show_normals;
    ui.toggle_normal_display->setState( show_normals, false );
}

void Visionaray::impl::set_show_tex_coords(bool show_tex_coords)
{
    dev_state->show_tex_coords = show_tex_coords;
    ui.toggle_tex_coord_display->setState( show_tex_coords, false );
}


//-------------------------------------------------------------------------------------------------
// Visionaray plugin
//

Visionaray::Visionaray()
    : impl_(new impl)
{
}

Visionaray::~Visionaray()
{
    opencover::cover->getObjectsRoot()->setNodeMask(impl_->objroot_node_mask);
    impl_->geode->removeDrawable(impl_->drawable_ptr);
    opencover::cover->getScene()->removeChild(impl_->geode);
}

bool Visionaray::init()
{
    using namespace osg;

    opencover::VRViewer::instance()->culling(false);

    std::cout << "Init Visionaray Plugin!!" << std::endl;

    impl_->init_ui();

    impl_->geode = new osg::Geode;
    impl_->geode->setName("Visionaray");
    impl_->geode->addDrawable(impl_->drawable_ptr);

    impl_->objroot_node_mask = opencover::cover->getObjectsRoot()->getNodeMask();

    opencover::cover->getScene()->addChild(impl_->geode);
    opencover::cover->getObjectsRoot()->setNodeMask
    (
        opencover::cover->getObjectsRoot()->getNodeMask()
     & ~opencover::VRViewer::instance()->getCullMask()
    );

    return true;
}

void Visionaray::preFrame()
{
}

void Visionaray::expandBoundingSphere(osg::BoundingSphere &bs)
{
    impl_->drawable_ptr->expandBoundingSphere(bs);
}

void Visionaray::key(int type, int key_sym, int /* mod */)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        switch (key_sym)
        {
        case '1':
            impl_->set_algorithm(Simple);
            break;

        case '2':
            impl_->set_algorithm(Whitted);
            break;

        case '3':
            impl_->set_algorithm(Pathtracing);
            break;
        }
    }
}

}} // namespace visionaray::cover

COVERPLUGIN(visionaray::cover::Visionaray)
