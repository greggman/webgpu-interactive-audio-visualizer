Make a simple WebGPU interactive audio visualzation webpage

1. Do not use any libraries
2. Make it so it plays a beat
3. Make it so keys act like a piano keyboard
4. Show a piano keyboard on screen and allow pointer events
   to control them so the user can use fingers on mobile
5. The audio must be generated in a compute shader given time
   and the latest input state
6. The audio must be sent to the Web Audio API via commincation
   with a Web Audio Worker
7. To be efficient, pre-mapAsync webgpu buffers so when you need
   a buffer you can immediately get one. After the buffer's use
   is submitted, use a promise to remap that buffer and put it
   in a list of pre-mapped buffers to be used next time
8. A separate compute shader should look at the audio output
   and visualize it. You can write a canvas texture by declaring
   it to have storage binding when you configure it. You should
   check for and if available, use the extension
   'bgra8unorm-storage' for efficiency on devices that prefer
   'bgra8unorm' format for canvas textures.
9. Label all WebGPU resources so error messages convey where
   they came from
10. Monitor uncaptureerror events from WebGPU so your testing
    will see webgpu errors
11. Use WebGPU best practices.

* Use typescript, gts, and esbuild at the latest compatible versions.
* include a github action to publish the build on github pages.

