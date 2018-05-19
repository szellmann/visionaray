// This file is distributed under the MIT license.
// See the LICENSE file for details.

#import <cassert>
#import <stdexcept>
#import <string>

#import <GL/glew.h>

#import <Cocoa/Cocoa.h>
#import <CoreVideo/CoreVideo.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>

#import "input/cocoa.h"
#import "input/key_event.h"
#import "input/mouse_event.h"
#import "viewer_cocoa.h"

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Event window
//

@interface graphics_view : NSView
{
@private
    // OpenGL context
    NSOpenGLContext* gl_context;

    // OpenGL pixel format
    NSOpenGLPixelFormat* gl_format;

    // Display link for window refresh
    CVDisplayLinkRef display_link;

    NSRect initial_bounds;
}
+ (NSOpenGLPixelFormat*)defaultPixelFormat;
- (void)registerDisplayLink;
- (id)initWithFrame:(NSRect)frame_rect
    pixel_format:(NSOpenGLPixelFormat*)format;
- (void) prepareOpenGL;
- (void)lockFocus;
- (void)update;
- (void)setOpenGLContext:(NSOpenGLContext*)context;
- (NSOpenGLContext*)openGLContext;
- (void)setPixelFormat:(NSOpenGLPixelFormat*)pixelFormat;
- (NSOpenGLPixelFormat*)pixelFormat;
- (void)viewDidMoveToWindow;
- (BOOL)acceptsFirstResponder;
- (BOOL)acceptsMouseMovedEvents;
- (void)drawRect: (NSRect)bounds;
- (void)dealloc;

- (void)render;
- (void)resize:(NSRect)bounds;

- (void)keyDown:(NSEvent*)event;

- (void)mouseDown:(NSEvent*)event;
- (void)mouseDragged:(NSEvent*)event;
- (void)mouseMoved:(NSEvent*)event;
- (void)mouseUp:(NSEvent*)event;

- (void)rightMouseDown:(NSEvent*)event;
- (void)rightMouseDragged:(NSEvent*)event;
- (void)rightMouseMoved:(NSEvent*)event;
- (void)rightMouseUp:(NSEvent*)event;

@property viewer_cocoa* viewer;
@end

@implementation graphics_view
+ (NSOpenGLPixelFormat*)defaultPixelFormat
{
    NSOpenGLPixelFormatAttribute attr[] = { 
        NSOpenGLPFAOpenGLProfile,
        NSOpenGLProfileVersionLegacy,//NSOpenGLProfileVersion3_2Core
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFADepthSize,
        24,
        0 // Last
        };  
 
    return [[NSOpenGLPixelFormat alloc]
        initWithAttributes: attr
        ];
}

- (void) registerDisplayLink
{
    // Regiser display link for continuous rendering

    gl_context = [[NSOpenGLContext alloc]
        initWithFormat: gl_format
        shareContext: nil
        ];

    GLint swap_int = 1;

    [gl_context setValues:&swap_int forParameter:NSOpenGLCPSwapInterval];

    CVDisplayLinkCreateWithActiveCGDisplays(&display_link);

    CVDisplayLinkSetOutputCallback(
            display_link,
            [](
                CVDisplayLinkRef   disp_link,
                CVTimeStamp const* now,
                CVTimeStamp const* output_time,
                CVOptionFlags      flags_in,
                CVOptionFlags*     flags_out,
                void*              display_link_context
                )
                -> CVReturn
            {
                VSNRAY_UNUSED(disp_link, now, output_time, flags_in, flags_out);

                graphics_view* view = static_cast<graphics_view*>(display_link_context);

                if (view->initial_bounds.size.width <= 0 || view->initial_bounds.size.height <= 0)
                {
                    return kCVReturnSuccess;
                }

                if ([view lockFocusIfCanDraw] == NO)
                {
                    return kCVReturnSuccess;
                }

                [view render];

                [view unlockFocus];

                return kCVReturnSuccess;
            },
            self
            );

    CGLContextObj cgl_context = [gl_context CGLContextObj];

    CGLPixelFormatObj cgl_format = [gl_format CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(
            display_link,
            cgl_context,
            cgl_format
            );

    CVDisplayLinkStart(display_link);
}

- (id)initWithFrame:(NSRect)frame_rect
     pixel_format:(NSOpenGLPixelFormat*)format
{
    self = [super initWithFrame: frame_rect];

    if (self != nil)
    {
        gl_format = [format retain];

        [[NSNotificationCenter defaultCenter] addObserver: self
            selector:@selector(_surfaceNeedsUpdate:)
            name: NSViewGlobalFrameDidChangeNotification
            object: self
            ];
    }

    return self;
}

- (void) _surfaceNeedsUpdate:(NSNotification*)notification
{
    VSNRAY_UNUSED(notification);

    [self update];
}

- (void) prepareOpenGL
{
    if ([self openGLContext] != nil)
    {
        return;
    }

    if ([self pixelFormat] == nil)
    {
        [self setPixelFormat:[graphics_view defaultPixelFormat]];
    }
    gl_context = [[NSOpenGLContext alloc]
        initWithFormat: gl_format
        shareContext: nil
        ];

    [gl_context makeCurrentContext];

    // GLEW

    GLenum error = glewInit();
    if (error != GLEW_OK)
    {
        std::string error_string("glewInit() failed: ");
        error_string.append(reinterpret_cast<char const*>(glewGetErrorString(error)));
        throw std::runtime_error(error_string);
    }
}

- (void)lockFocus
{
    [super lockFocus];

    if ([gl_context view] != self)
    {
        [gl_context setView: self];
    }

    [gl_context makeCurrentContext];
}

- (void)update
{
    [gl_context update];
}

- (void)setOpenGLContext:(NSOpenGLContext*)context
{
    gl_context = context;
}

- (NSOpenGLContext*)openGLContext
{
    return gl_context;
}

- (void)setPixelFormat:(NSOpenGLPixelFormat*)pixelFormat
{
    gl_format = pixelFormat;
}

- (NSOpenGLPixelFormat*)pixelFormat
{
    return gl_format;
}

- (void)viewDidMoveToWindow
{
    [super viewDidMoveToWindow];

    if ([self window] == nil)
    {
        [gl_context clearDrawable];
    }
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (BOOL)acceptsMouseMovedEvents
{
    return YES;
}

- (void)drawRect: (NSRect)bounds
{
    [gl_context makeCurrentContext];

    if (bounds.size.width != initial_bounds.size.width
     || bounds.size.height != initial_bounds.size.height)
    {
        [self resize: bounds];
    }

    [self render];
}

- (void)dealloc
{
    CVDisplayLinkRelease(display_link);

    [super dealloc];
}

- (void)render
{
    [gl_context makeCurrentContext];

    _viewer->call_on_display();

    [gl_context flushBuffer];
}

- (void)resize: (NSRect)bounds
{
    _viewer->call_on_resize(bounds.size.width, bounds.size.height);
    [gl_context update];
    initial_bounds = bounds;
}

- (void)keyDown:(NSEvent*)event
{
    [self setNeedsDisplay: YES];

    NSString* str = [event characters];
    char const* chars = [str UTF8String];

    assert(chars);

    key_event e(keyboard::KeyPress, keyboard::map_cocoa_key(chars[0]));

    _viewer->call_on_key_press(e);
}

- (void)mouseDown:(NSEvent*)event
{
    [self setNeedsDisplay:YES];

    NSEventType et = [event type];
    CGPoint epos = [event locationInWindow];
    mouse::pos p = { epos.x, self.frame.size.height - epos.y - 1 };

    auto b = mouse::map_cocoa_button(et);
    NSUInteger mf = [event modifierFlags] & NSEventModifierFlagDeviceIndependentFlagsMask;
    auto m = keyboard::map_cocoa_modifiers(mf);

    mouse_event e(mouse::ButtonDown, p, b, m);

    _viewer->call_on_mouse_down(e);
}

- (void)mouseDragged:(NSEvent*)event
{
    [self setNeedsDisplay:YES];

    NSEventType et = [event type];
    CGPoint epos = [event locationInWindow];
    mouse::pos p = { epos.x, self.frame.size.height - epos.y - 1 };

    auto b = mouse::map_cocoa_button(et);
    NSUInteger mf = [event modifierFlags] & NSEventModifierFlagDeviceIndependentFlagsMask;
    auto m = keyboard::map_cocoa_modifiers(mf);

    mouse_event e(mouse::Move, p, b, m);

    _viewer->call_on_mouse_move(e);
}

- (void)mouseMoved:(NSEvent*)event
{
    [self setNeedsDisplay:YES];

    NSEventType et = [event type];
    CGPoint epos = [event locationInWindow];
    mouse::pos p = { epos.x, self.frame.size.height - epos.y - 1 };

    auto b = mouse::map_cocoa_button(et);
    NSUInteger mf = [event modifierFlags] & NSEventModifierFlagDeviceIndependentFlagsMask;
    auto m = keyboard::map_cocoa_modifiers(mf);

    mouse_event e(mouse::Move, p, b, m);

    _viewer->call_on_mouse_move(e);
}

- (void)mouseUp:(NSEvent*)event
{
    [self setNeedsDisplay:YES];

    NSEventType et = [event type];
    CGPoint epos = [event locationInWindow];
    mouse::pos p = { epos.x, self.frame.size.height - epos.y - 1 };

    auto b = mouse::map_cocoa_button(et);
    NSUInteger mf = [event modifierFlags] & NSEventModifierFlagDeviceIndependentFlagsMask;
    auto m = keyboard::map_cocoa_modifiers(mf);

    mouse_event e(mouse::Move, p, b, m);

    _viewer->call_on_mouse_up(e);
}

- (void)rightMouseDown:(NSEvent*)event
{
    [self mouseDown:event];
}

- (void)rightMouseDragged:(NSEvent*)event
{
    [self mouseDragged:event];
}

- (void)rightMouseMoved:(NSEvent*)event
{
    [self mouseMoved:event];
}

- (void)rightMouseUp:(NSEvent*)event
{
    [self mouseUp:event];
}

@end

//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct viewer_cocoa::impl
{
    viewer_cocoa* viewer = nullptr;
    NSWindow* window = nullptr;
    graphics_view* view = nullptr;

    // Object that maintains reference count to other Cocoa objects
    NSAutoreleasePool* pool = nullptr;
};


//-------------------------------------------------------------------------------------------------
// viewer_cocoa
//

viewer_cocoa::viewer_cocoa(int width, int height, char const* window_title)
    : viewer_base(width, height, window_title)
    , impl_(new impl)
{
}

viewer_cocoa::~viewer_cocoa()
{
}

void viewer_cocoa::init(int argc, char** argv)
{
    viewer_base::init(argc, argv);

    // Init Cocoa reference counting
    impl_->pool = [[NSAutoreleasePool alloc] init];

    // Init Cocoa application
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    // Create window
    NSRect rect = NSMakeRect(
            0.0f,
            0.0f,
            static_cast<float>(width()),
            static_cast<float>(height())
            );

    impl_->window = [[[NSWindow alloc]
            initWithContentRect: rect
            styleMask: NSWindowStyleMaskTitled
                     | NSWindowStyleMaskResizable
                     | NSWindowStyleMaskMiniaturizable
                     | NSWindowStyleMaskClosable
            backing: NSBackingStoreBuffered
            defer: NO
            ] autorelease];

    NSString* nstitle = [[[NSString alloc]
            initWithCString: window_title()
            encoding: NSUTF8StringEncoding
            ] autorelease];

    [impl_->window setTitle: nstitle];

    // Init OpenGL view

    impl_->view = [[graphics_view alloc] initWithFrame: rect];
    [impl_->view prepareOpenGL];
    [impl_->view setHidden: NO];
    [impl_->view setNeedsDisplay: YES];
    [impl_->view setViewer: this];
    [impl_->window setContentView: impl_->view];

    // Show window
    [impl_->window makeKeyAndOrderFront: NSApp];
}

void viewer_cocoa::event_loop()
{
    // Bring to front
    [NSApp activateIgnoringOtherApps: YES];

    // Init continuous rendering
    [impl_->view registerDisplayLink];

    // Run application
    [NSApp run];
}

void viewer_cocoa::resize(int width, int height)
{
    viewer_base::resize(width, height);
}

void viewer_cocoa::swap_buffers()
{
}

void viewer_cocoa::toggle_full_screen()
{
    [impl_->window toggleFullScreen: nil];
}

void viewer_cocoa::quit()
{
    // Quit application
    [NSApp terminate: nil];
}

void viewer_cocoa::call_on_display()
{
    on_display();
}

void viewer_cocoa::call_on_key_press(key_event const& event)
{
    on_key_press(event);
}

void viewer_cocoa::call_on_mouse_down(mouse_event const& event)
{
    on_mouse_down(event);
}

void viewer_cocoa::call_on_mouse_move(mouse_event const& event)
{
    on_mouse_move(event);
}

void viewer_cocoa::call_on_mouse_up(mouse_event const& event)
{
    on_mouse_up(event);
}

void viewer_cocoa::call_on_resize(int width, int height)
{
    on_resize(width, height);
}
