'use strict';

const fs = require('fs');
const path = require('path');

const srcDir = 'public';
const distDir = 'dist';

fs.mkdirSync(distDir, {recursive: true});

for (const entry of fs.readdirSync(srcDir, {withFileTypes: true})) {
  if (entry.isFile()) {
    fs.copyFileSync(
      path.join(srcDir, entry.name),
      path.join(distDir, entry.name),
    );
  }
}
