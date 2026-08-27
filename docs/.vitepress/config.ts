import { defineConfig } from 'vitepress'
import { generateSidebar } from 'vitepress-sidebar'

const autoSidebar = generateSidebar({
  documentRootPath: '/docs/',
  collapsed: true,
});

// 在自动生成的侧边栏后面追加自定义外部链接分组
// JavaScript/TypeScript 中的展开运算符...，把 autoSidebar 数组里的所有元素"展开"放到这里
const sidebar = [
  ...autoSidebar,
  {
    text: "外部链接",
    collapsed: true,
    items: [
      { text: 'my blog', link: 'https://hankin2015.github.io' },
      { text: 'github', link: 'https://github.com/hankin2015' }
    ]
  }
]

export default defineConfig({
  lang: 'zh-CN',
  title: 'hankin',
  description: 'Simple, light-weight and easy-to-use asynchronous components',
  base: '/Machine_to_DeepingLearning/',
  lastUpdated: true,
  ignoreDeadLinks: false,
  outDir: "public",
  locales: {
    "/docs.en/": {
      lang: 'en-US',
      title: 'Machine_to_DeepingLearning',
      description: 'Simple, light-weight and easy-to-use asynchronous components',
    },
    "/docs.cn/": {
      lang: 'zh-CN',
      title: 'Machine_to_DeepingLearning',
      description: 'Simple, light-weight and easy-to-use asynchronous components',
    },
  },
  head: [],

  themeConfig: {
    nav: nav(),

    sidebar: {
      "/docs.en/": sidebar,
      '/docs.cn/': sidebar,
    },

    socialLinks: [
      {icon: 'github', link: 'https://github.com/alibaba/async_simple'}
    ],

    footer: {
      message: 'This website is released under the MIT License.',
      copyright: 'Copyright © 2026 hankin2015 contributors'
    },

    editLink: {
      pattern: 'https://github.com/alibaba/async_simple/edit/main/docs/:path'
    }
  }
})

function nav() {
  return [
    {text: 'Guide', link: '/docs.en/GetStarted', activeMatch: '/guide/'},
    {
      text: "Language",
      items: [
        {
          text: "English", link: "/docs.en/GetStarted"
        },
        {
          text: "简体中文", link: '/docs.cn/GetStarted'
        }
      ]
    },
    {
      text: 'Github Issues',
      link: 'https://github.com/alibaba/async_simple/issues'
    }
  ]
}
