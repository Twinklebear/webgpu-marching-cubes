# WebGPU Webpack + TypeScript Starter

A starter template for [WebGPU](https://gpuweb.github.io/gpuweb/) and TypeScript development that uses [webpack](https://webpack.js.org/) to build the code
and manage dependencies. You'll also need [node.js](https://nodejs.org/) installed for package management
and running tasks.

## Getting Started

After cloning the repo run

```
npm install
```

To install webpack, then you can run the serve task and point your browser to `localhost:8080`:

```
npm run serve
```

Where you should see the page shown below.

To deploy your application, run:

```
npm run deploy
```

Then you can copy the content of the `dist/` directory to your webserver. You can build a development
distribution by running `npm run build`.

If everything's working you should see this page in your browser when you run the application:

<img width="1147" alt="Expected output of the starter template" src="https://user-images.githubusercontent.com/1522476/133895532-03f84dbd-bb3a-4c74-ab9d-fd74506c3a74.png">
